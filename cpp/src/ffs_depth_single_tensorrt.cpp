#include "ffs_depth_single_tensorrt.hpp"

#include "ffs_gwc_plugin.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ffs_depth {

namespace cuda {
extern "C" {

void ffsCudaPreprocessRGBToCHW(
    const uint8_t* d_rgb, float* d_chw,
    int src_h, int src_w, int dst_h, int dst_w, cudaStream_t s);

void ffsCudaResizeUniformAndPad(
    const uint8_t* d_rgb, float* d_chw,
    int src_h, int src_w, int scaled_h, int scaled_w,
    int dst_h, int dst_w, cudaStream_t s);

void ffsCudaCropDisparity(
    const float* d_src, float* d_dst,
    int src_h, int src_w, int dst_h, int dst_w, cudaStream_t s);

void ffsCudaUpsampleDisparity(
    const float* d_src, float* d_dst,
    int src_w, int src_h, int dst_w, int dst_h,
    float disp_scale, cudaStream_t s);

void ffsCudaClampDisparity(
    float* d_disp, int count, float min_val, cudaStream_t s);

void ffsCudaDispToDepth(
    const float* d_disp, float* d_depth_m,
    int height, int width, float fx, float baseline_m, cudaStream_t s);

}  // extern "C"
}  // namespace cuda

namespace {

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TRT] " << msg << std::endl;
        }
    }
};

TrtLogger g_trt_logger;

size_t elementSize(nvinfer1::DataType dt) {
    switch (dt) {
        case nvinfer1::DataType::kFLOAT: return 4;
        case nvinfer1::DataType::kHALF:  return 2;
        case nvinfer1::DataType::kINT8:  return 1;
        case nvinfer1::DataType::kINT32: return 4;
        default: return 4;
    }
}

void cudaMallocChecked(void** ptr, size_t bytes, const char* what) {
    cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[FFS single] cudaMalloc failed for ") +
                                 what + " (" + std::to_string(bytes) + " bytes): " +
                                 cudaGetErrorString(err));
    }
}

bool hasTensor(nvinfer1::ICudaEngine& engine, const char* name) {
    for (int32_t i = 0; i < engine.getNbIOTensors(); ++i) {
        if (std::string(engine.getIOTensorName(i)) == name) return true;
    }
    return false;
}

std::string findSingleYamlConfig(const std::string& engine_dir) {
    namespace fs = std::filesystem;

    std::vector<fs::path> yaml_files;
    std::error_code ec;
    fs::directory_iterator it(engine_dir, ec);
    if (ec) {
        throw std::runtime_error("[FFS single] Cannot list engine directory: " +
                                 engine_dir + " (" + ec.message() + ")");
    }

    for (const auto& entry : it) {
        if (ec) {
            throw std::runtime_error("[FFS single] Cannot list engine directory: " +
                                     engine_dir + " (" + ec.message() + ")");
        }
        if (!entry.is_regular_file()) continue;

        std::string ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
        if (ext == ".yaml" || ext == ".yml") {
            yaml_files.push_back(entry.path());
        }
    }

    if (yaml_files.empty()) {
        throw std::runtime_error("[FFS single] No YAML config found in: " + engine_dir);
    }
    if (yaml_files.size() > 1) {
        std::ostringstream oss;
        oss << "[FFS single] Expected exactly one YAML config in " << engine_dir
            << ", found " << yaml_files.size() << ":";
        for (const auto& path : yaml_files) {
            oss << " " << path.string();
        }
        throw std::runtime_error(oss.str());
    }

    return yaml_files.front().string();
}

std::string trim(std::string s) {
    const char* ws = " \t\r\n";
    const size_t start = s.find_first_not_of(ws);
    if (start == std::string::npos) return "";
    const size_t end = s.find_last_not_of(ws);
    return s.substr(start, end - start + 1);
}

std::vector<int> parseInts(std::string s) {
    for (char& c : s) {
        if (!(c >= '0' && c <= '9')) c = ' ';
    }
    std::stringstream ss(s);
    std::vector<int> values;
    int v = 0;
    while (ss >> v) values.push_back(v);
    return values;
}

bool parseBool(std::string s) {
    s = trim(s);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return s == "true" || s == "1" || s == "yes" || s == "on";
}

}  // namespace

FFSSingleEngineInference::FFSSingleEngineInference(const std::string& engine_dir) {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[FFS single] cudaStreamCreate failed: ") +
                                 cudaGetErrorString(err));
    }
    try {
        loadConfig(findSingleYamlConfig(engine_dir));
        if (!registerFFSGWCPlugin()) {
            throw std::runtime_error("[FFS single] failed to register FFSGWCVolume plugin");
        }
        runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
        if (!runtime_) {
            throw std::runtime_error("[FFS single] createInferRuntime returned null");
        }
        loadEngine(engine_dir + "/fast_foundationstereo.engine");
        allocateBuffers();
    } catch (...) {
        freeDeviceBuffers();
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        throw;
    }
}

FFSSingleEngineInference::~FFSSingleEngineInference() {
    freeDeviceBuffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void FFSSingleEngineInference::freeDeviceBuffers() {
    auto free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };
    free(reinterpret_cast<void*&>(d_left_));
    free(reinterpret_cast<void*&>(d_right_));
    free(reinterpret_cast<void*&>(d_disp_));
    free(reinterpret_cast<void*&>(d_disp_cropped_));
    free(reinterpret_cast<void*&>(d_disp_for_depth_));
    depth_alloc_pixels_ = 0;
}

void FFSSingleEngineInference::loadConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) throw std::runtime_error("[FFS single] Cannot open config: " + path);

    std::string line;
    int image_index = -1;
    while (std::getline(f, line)) {
        const size_t comment = line.find('#');
        if (comment != std::string::npos) line = line.substr(0, comment);
        line = trim(line);
        if (line.empty()) continue;

        if (image_index >= 0) {
            if (line.rfind("-", 0) == 0) {
                const auto values = parseInts(line);
                if (!values.empty()) {
                    if (image_index == 0) config_.image_height = values[0];
                    if (image_index == 1) config_.image_width = values[0];
                    ++image_index;
                    if (image_index >= 2) image_index = -1;
                    continue;
                }
            }
            image_index = -1;
        }

        const size_t colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon));
        const std::string value = trim(line.substr(colon + 1));

        if (key == "image_size") {
            const auto values = parseInts(value);
            if (values.size() >= 2) {
                config_.image_height = values[0];
                config_.image_width = values[1];
            } else {
                image_index = 0;
            }
        } else if (key == "max_disp") {
            const auto values = parseInts(value);
            if (!values.empty()) config_.max_disp = values[0];
        } else if (key == "cv_group") {
            const auto values = parseInts(value);
            if (!values.empty()) config_.cv_group = values[0];
        } else if (key == "valid_iters") {
            const auto values = parseInts(value);
            if (!values.empty()) config_.valid_iters = values[0];
        } else if (key == "normalize") {
            config_.normalize = parseBool(value);
        }
    }
}

void FFSSingleEngineInference::loadEngine(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) throw std::runtime_error("[FFS single] Cannot open engine: " + path);

    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);

    engine_.reset(runtime_->deserializeCudaEngine(buf.data(), sz));
    if (!engine_) throw std::runtime_error("[FFS single] Deserialize failed: " + path);

    context_.reset(engine_->createExecutionContext());
    if (!context_) throw std::runtime_error("[FFS single] Context creation failed: " + path);

    if (!hasTensor(*engine_, "left") || !hasTensor(*engine_, "right") || !hasTensor(*engine_, "disp")) {
        throw std::runtime_error("[FFS single] Engine must expose tensors named left, right, and disp");
    }
}

void FFSSingleEngineInference::allocateBuffers() {
    const size_t H = static_cast<size_t>(config_.image_height);
    const size_t W = static_cast<size_t>(config_.image_width);
    const size_t input_bytes = 3 * H * W * sizeof(float);

    cudaMallocChecked(reinterpret_cast<void**>(&d_left_), input_bytes, "d_left_");
    cudaMallocChecked(reinterpret_cast<void**>(&d_right_), input_bytes, "d_right_");

    if (engine_->getTensorDataType("disp") != nvinfer1::DataType::kFLOAT) {
        throw std::runtime_error("[FFS single] disp tensor must be FP32");
    }
    const size_t disp_bytes = H * W * sizeof(float);
    cudaMallocChecked(reinterpret_cast<void**>(&d_disp_), disp_bytes, "d_disp_");
    cudaMallocChecked(reinterpret_cast<void**>(&d_disp_cropped_),
                      H * W * sizeof(float), "d_disp_cropped_");

    context_->setTensorAddress("left", d_left_);
    context_->setTensorAddress("right", d_right_);
    context_->setTensorAddress("disp", d_disp_);
}

void FFSSingleEngineInference::preprocessRGBGPU(
    const uint8_t* d_rgb, int src_h, int src_w, float* d_output) {
    const int mH = config_.image_height;
    const int mW = config_.image_width;
    if (src_h == mH && src_w == mW) {
        scaled_h_ = mH;
        scaled_w_ = mW;
        cuda::ffsCudaPreprocessRGBToCHW(d_rgb, d_output, src_h, src_w, mH, mW, stream_);
    } else {
        const float scale = std::min(static_cast<float>(mW) / src_w,
                                     static_cast<float>(mH) / src_h);
        scaled_w_ = std::max(1, static_cast<int>(std::round(src_w * scale)));
        scaled_h_ = std::max(1, static_cast<int>(std::round(src_h * scale)));
        cuda::ffsCudaResizeUniformAndPad(
            d_rgb, d_output, src_h, src_w, scaled_h_, scaled_w_, mH, mW, stream_);
    }
}

void FFSSingleEngineInference::infer(
    const uint8_t* d_left_rgb, const uint8_t* d_right_rgb,
    int input_h, int input_w,
    float* d_disp_out) {
    if (!d_left_rgb || !d_right_rgb || !d_disp_out) {
        throw std::runtime_error("[FFS single] infer: null device pointer");
    }
    if (input_h <= 0 || input_w <= 0) {
        throw std::runtime_error("[FFS single] infer: input dimensions must be positive");
    }

    const int mH = config_.image_height;
    const int mW = config_.image_width;
    const bool needs_resize = (input_h != mH || input_w != mW);

    preprocessRGBGPU(d_left_rgb, input_h, input_w, d_left_);
    preprocessRGBGPU(d_right_rgb, input_h, input_w, d_right_);

    if (!needs_resize) {
        context_->setTensorAddress("disp", d_disp_out);
    }
    if (!context_->enqueueV3(stream_)) {
        throw std::runtime_error("[FFS single] enqueue failed");
    }

    if (needs_resize) {
        cuda::ffsCudaClampDisparity(d_disp_, mH * mW, 0.0f, stream_);
        cuda::ffsCudaCropDisparity(d_disp_, d_disp_cropped_,
                                   mH, mW, scaled_h_, scaled_w_, stream_);
        const float disp_scale = static_cast<float>(input_w) / scaled_w_;
        cuda::ffsCudaUpsampleDisparity(d_disp_cropped_, d_disp_out,
                                       scaled_w_, scaled_h_, input_w, input_h,
                                       disp_scale, stream_);
    } else {
        cuda::ffsCudaClampDisparity(d_disp_out, mH * mW, 0.0f, stream_);
        context_->setTensorAddress("disp", d_disp_);
    }
}

void FFSSingleEngineInference::dispToDepth(
    const float* d_disp,
    int height, int width,
    float fx, float baseline_m,
    float* d_depth_out) {
    cuda::ffsCudaDispToDepth(d_disp, d_depth_out,
                             height, width, fx, baseline_m, stream_);
}

void FFSSingleEngineInference::inferDepth(
    const uint8_t* d_left_rgb, const uint8_t* d_right_rgb,
    int input_h, int input_w,
    float fx, float baseline_m,
    float* d_depth_out) {
    if (input_h <= 0 || input_w <= 0) {
        throw std::runtime_error("[FFS single] inferDepth: input dimensions must be positive");
    }
    const size_t num_pixels = static_cast<size_t>(input_h) * static_cast<size_t>(input_w);
    if (static_cast<int64_t>(num_pixels) > depth_alloc_pixels_) {
        if (d_disp_for_depth_) {
            cudaFree(d_disp_for_depth_);
            d_disp_for_depth_ = nullptr;
        }
        cudaMallocChecked(reinterpret_cast<void**>(&d_disp_for_depth_),
                          num_pixels * sizeof(float), "d_disp_for_depth_");
        depth_alloc_pixels_ = static_cast<int64_t>(num_pixels);
    }
    infer(d_left_rgb, d_right_rgb, input_h, input_w, d_disp_for_depth_);
    cuda::ffsCudaDispToDepth(d_disp_for_depth_, d_depth_out,
                             input_h, input_w, fx, baseline_m, stream_);
}

void FFSSingleEngineInference::sync() {
    cudaStreamSynchronize(stream_);
}

}  // namespace ffs_depth
