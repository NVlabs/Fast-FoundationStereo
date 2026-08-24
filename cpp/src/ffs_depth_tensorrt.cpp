#include "ffs_depth_tensorrt.hpp"

#include <cuda_fp16.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace ffs_depth {

// Forward declarations of CUDA kernel wrappers (defined in depth_kernels.cu)
namespace cuda {
extern "C" {

void ffsCudaBuildGWCVolumeMixed(
    const float* d_fl, const float* d_fr, __half* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s);

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
        if (severity <= Severity::kWARNING)
            std::cerr << "[TRT] " << msg << std::endl;
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
        throw std::runtime_error(std::string("[FFS] cudaMalloc failed for ") + what +
                                 " (" + std::to_string(bytes) + " bytes): " +
                                 cudaGetErrorString(err));
    }
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

}  // anonymous namespace

// =========================================================================
// Construction / destruction
// =========================================================================

FFSDepthInference::FFSDepthInference(const std::string& engine_dir) {
    cudaError_t err = cudaStreamCreate(&stream_);
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("[FFS] cudaStreamCreate failed: ") +
                                 cudaGetErrorString(err));
    }
    try {
        loadConfig(engine_dir + "/onnx.yaml");
        runtime_.reset(nvinfer1::createInferRuntime(g_trt_logger));
        if (!runtime_) {
            throw std::runtime_error("[FFS] createInferRuntime returned null");
        }
        loadEngine(engine_dir + "/feature_runner.engine", feature_engine_, feature_context_);
        loadEngine(engine_dir + "/post_runner.engine", post_engine_, post_context_);
        allocateBuffers();
    } catch (...) {
        // Best-effort cleanup of anything allocated so far. Buffers may be partially
        // populated; freeDeviceBuffers() is a no-op for null pointers.
        freeDeviceBuffers();
        if (stream_) {
            cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
        throw;
    }
}


FFSDepthInference::~FFSDepthInference() {
    freeDeviceBuffers();
    if (stream_) {
        cudaStreamDestroy(stream_);
        stream_ = nullptr;
    }
}

void FFSDepthInference::freeDeviceBuffers() {
    auto free = [](void*& p) { if (p) { cudaFree(p); p = nullptr; } };

    free(reinterpret_cast<void*&>(d_left_));
    free(reinterpret_cast<void*&>(d_right_));
    free(reinterpret_cast<void*&>(d_feat_left_04_));
    free(reinterpret_cast<void*&>(d_feat_left_08_));
    free(reinterpret_cast<void*&>(d_feat_left_16_));
    free(reinterpret_cast<void*&>(d_feat_left_32_));
    free(reinterpret_cast<void*&>(d_feat_right_04_));
    free(reinterpret_cast<void*&>(d_stem_2x_));
    free(reinterpret_cast<void*&>(d_gwc_volume_));
    free(reinterpret_cast<void*&>(d_disp_));
    free(reinterpret_cast<void*&>(d_disp_cropped_));
    free(reinterpret_cast<void*&>(d_disp_for_depth_));
    depth_alloc_pixels_ = 0;
}

// =========================================================================
// Config / engine loading
// =========================================================================

void FFSDepthInference::loadConfig(const std::string& path) {
    std::ifstream f(path);
    if (!f.good()) throw std::runtime_error("[FFS] Cannot open config: " + path);

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

    gwc_disp_levels_ = config_.max_disp / 4;
}

void FFSDepthInference::loadEngine(
    const std::string& path,
    std::unique_ptr<nvinfer1::ICudaEngine>& engine,
    std::unique_ptr<nvinfer1::IExecutionContext>& ctx)
{
    std::ifstream f(path, std::ios::binary);
    if (!f.good()) throw std::runtime_error("[FFS] Cannot open engine: " + path);

    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(sz);
    f.read(buf.data(), sz);

    engine.reset(runtime_->deserializeCudaEngine(buf.data(), sz));
    if (!engine) throw std::runtime_error("[FFS] Deserialize failed: " + path);

    ctx.reset(engine->createExecutionContext());
    if (!ctx) throw std::runtime_error("[FFS] Context creation failed: " + path);
}

// =========================================================================
// Buffer allocation
// =========================================================================

void FFSDepthInference::allocateBuffers() {
    const size_t H = static_cast<size_t>(config_.image_height);
    const size_t W = static_cast<size_t>(config_.image_width);
    const size_t input_bytes = 3 * H * W * sizeof(float);

    cudaMallocChecked(reinterpret_cast<void**>(&d_left_),  input_bytes, "d_left_");
    cudaMallocChecked(reinterpret_cast<void**>(&d_right_), input_bytes, "d_right_");

    allocateFeatureBuffers();
    allocatePostBuffers();
}

void FFSDepthInference::allocateFeatureBuffers() {
    auto getDims = [&](const char* name) {
        auto d = feature_engine_->getTensorShape(name);
        std::vector<int> v(d.nbDims);
        for (int i = 0; i < d.nbDims; ++i) v[i] = d.d[i];
        return v;
    };

    auto allocTensor = [&](const char* name, const std::vector<int>& dims) -> float* {
        auto dt = feature_engine_->getTensorDataType(name);
        size_t bytes = elementSize(dt);
        for (int d : dims) {
            if (d <= 0) {
                throw std::runtime_error(std::string("[FFS] tensor '") + name +
                                         "' has non-positive dimension");
            }
            bytes *= static_cast<size_t>(d);
        }
        void* p = nullptr;
        cudaMallocChecked(&p, bytes, name);
        return static_cast<float*>(p);
    };

    feat_04_dims_  = getDims("features_left_04");
    feat_08_dims_  = getDims("features_left_08");
    feat_16_dims_  = getDims("features_left_16");
    feat_32_dims_  = getDims("features_left_32");
    stem_2x_dims_  = getDims("stem_2x");

    d_feat_left_04_  = allocTensor("features_left_04",  feat_04_dims_);
    d_feat_left_08_  = allocTensor("features_left_08",  feat_08_dims_);
    d_feat_left_16_  = allocTensor("features_left_16",  feat_16_dims_);
    d_feat_left_32_  = allocTensor("features_left_32",  feat_32_dims_);
    d_feat_right_04_ = allocTensor("features_right_04", feat_04_dims_);
    d_stem_2x_       = allocTensor("stem_2x",           stem_2x_dims_);

    feature_context_->setTensorAddress("left",               d_left_);
    feature_context_->setTensorAddress("right",              d_right_);
    feature_context_->setTensorAddress("features_left_04",   d_feat_left_04_);
    feature_context_->setTensorAddress("features_left_08",   d_feat_left_08_);
    feature_context_->setTensorAddress("features_left_16",   d_feat_left_16_);
    feature_context_->setTensorAddress("features_left_32",   d_feat_left_32_);
    feature_context_->setTensorAddress("features_right_04",  d_feat_right_04_);
    feature_context_->setTensorAddress("stem_2x",            d_stem_2x_);
}

void FFSDepthInference::allocatePostBuffers() {
    const size_t H  = static_cast<size_t>(config_.image_height);
    const size_t W  = static_cast<size_t>(config_.image_width);
    const size_t H4 = H / 4;
    const size_t W4 = W / 4;

    const auto gwc_dt = post_engine_->getTensorDataType("gwc_volume");
    gwc_fp16_ = (gwc_dt == nvinfer1::DataType::kHALF);
    const size_t gwc_elem = gwc_fp16_ ? sizeof(__half) : sizeof(float);
    const size_t gwc_bytes =
        static_cast<size_t>(config_.cv_group) *
        static_cast<size_t>(gwc_disp_levels_) * H4 * W4 * gwc_elem;
    cudaMallocChecked(reinterpret_cast<void**>(&d_gwc_volume_), gwc_bytes, "gwc_volume");

    const auto disp_dt = post_engine_->getTensorDataType("disp");
    const size_t disp_bytes = H * W * elementSize(disp_dt);
    cudaMallocChecked(reinterpret_cast<void**>(&d_disp_), disp_bytes, "d_disp_");

    const size_t disp_cropped_bytes = H * W * sizeof(float);
    cudaMallocChecked(reinterpret_cast<void**>(&d_disp_cropped_),
                      disp_cropped_bytes, "d_disp_cropped_");

    post_context_->setTensorAddress("features_left_04",  d_feat_left_04_);
    post_context_->setTensorAddress("features_left_08",  d_feat_left_08_);
    post_context_->setTensorAddress("features_left_16",  d_feat_left_16_);
    post_context_->setTensorAddress("features_left_32",  d_feat_left_32_);
    post_context_->setTensorAddress("features_right_04", d_feat_right_04_);
    post_context_->setTensorAddress("stem_2x",           d_stem_2x_);
    post_context_->setTensorAddress("gwc_volume",        d_gwc_volume_);
    post_context_->setTensorAddress("disp",              d_disp_);
}

// =========================================================================
// Pre-processing
// =========================================================================

void FFSDepthInference::preprocessRGBGPU(
    const uint8_t* d_rgb, int src_h, int src_w, float* d_output)
{
    const int mH = config_.image_height;
    const int mW = config_.image_width;

    if (src_h == mH && src_w == mW) {
        scaled_h_ = mH;
        scaled_w_ = mW;
        cuda::ffsCudaPreprocessRGBToCHW(
            d_rgb, d_output, src_h, src_w, mH, mW, stream_);
    } else {
        const float scale = std::min(static_cast<float>(mW) / src_w,
                                     static_cast<float>(mH) / src_h);
        scaled_w_ = std::max(1, static_cast<int>(std::round(src_w * scale)));
        scaled_h_ = std::max(1, static_cast<int>(std::round(src_h * scale)));
        cuda::ffsCudaResizeUniformAndPad(
            d_rgb, d_output, src_h, src_w, scaled_h_, scaled_w_, mH, mW, stream_);
    }
}

// =========================================================================
// Pipeline stages
// =========================================================================

void FFSDepthInference::runFeatureRunner() {
    if (!feature_context_->enqueueV3(stream_))
        throw std::runtime_error("[FFS] feature_runner failed");
}

void FFSDepthInference::buildGWCVolume() {
    int B = feat_04_dims_[0];
    int C = feat_04_dims_[1];
    int H = feat_04_dims_[2];
    int W = feat_04_dims_[3];

    // Only mixed precision path is used (FP32 features -> FP16 GWC volume)
    cuda::ffsCudaBuildGWCVolumeMixed(
        d_feat_left_04_, d_feat_right_04_,
        reinterpret_cast<__half*>(d_gwc_volume_),
        B, C, H, W, gwc_disp_levels_, config_.cv_group, config_.normalize, stream_);

}

void FFSDepthInference::runPostRunner() {
    if (!post_context_->enqueueV3(stream_))
        throw std::runtime_error("[FFS] post_runner failed");
}


// =========================================================================
// Public inference entry point
// =========================================================================

void FFSDepthInference::infer(
    const uint8_t* d_left_rgb, const uint8_t* d_right_rgb,
    int input_h, int input_w,
    float* d_disp_out)
{
    if (!d_left_rgb || !d_right_rgb || !d_disp_out) {
        throw std::runtime_error("[FFS] infer: null device pointer");
    }
    if (input_h <= 0 || input_w <= 0) {
        throw std::runtime_error("[FFS] infer: input dimensions must be positive");
    }
    const int mH = config_.image_height;
    const int mW = config_.image_width;
    const bool needs_resize = (input_h != mH || input_w != mW);
    preprocessRGBGPU(d_left_rgb,  input_h, input_w, d_left_);
    preprocessRGBGPU(d_right_rgb, input_h, input_w, d_right_);
    runFeatureRunner();
    buildGWCVolume();
    if (!needs_resize) {
        post_context_->setTensorAddress("disp", d_disp_out);
    }
    runPostRunner();
    if (needs_resize) {
        cuda::ffsCudaClampDisparity(d_disp_, mH * mW, 0.0f, stream_);
        // Crop padding: model (mW x mH) -> scaled (scaled_w_ x scaled_h_)
        cuda::ffsCudaCropDisparity(
            d_disp_, d_disp_cropped_,
            mH, mW, scaled_h_, scaled_w_, stream_);
        // Upsample to input resolution with disparity scale correction
        float disp_scale = static_cast<float>(input_w) / scaled_w_;
        cuda::ffsCudaUpsampleDisparity(
            d_disp_cropped_, d_disp_out,
            scaled_w_, scaled_h_, input_w, input_h,
            disp_scale, stream_);
    } else {
        cuda::ffsCudaClampDisparity(d_disp_out, mH * mW, 0.0f, stream_);
        post_context_->setTensorAddress("disp", d_disp_);
    }
}

void FFSDepthInference::sync() {
    cudaStreamSynchronize(stream_);
}

void FFSDepthInference::dispToDepth(
    const float* d_disp,
    int height, int width,
    float fx, float baseline_m,
    float* d_depth_out)
{
    cuda::ffsCudaDispToDepth(d_disp, d_depth_out,
                             height, width, fx, baseline_m, stream_);
}

void FFSDepthInference::inferDepth(
    const uint8_t* d_left_rgb, const uint8_t* d_right_rgb,
    int input_h, int input_w,
    float fx, float baseline_m,
    float* d_depth_out)
{
    if (!d_left_rgb || !d_right_rgb || !d_depth_out) {
        throw std::runtime_error("[FFS] inferDepth: null device pointer");
    }
    if (input_h <= 0 || input_w <= 0) {
        throw std::runtime_error("[FFS] inferDepth: input dimensions must be positive");
    }
    const size_t num_pixels = static_cast<size_t>(input_h) * static_cast<size_t>(input_w);

    if (static_cast<int64_t>(num_pixels) > depth_alloc_pixels_) {
        // cudaFree is implicitly stream-ordered: it waits for prior work on the
        // device before reclaiming the allocation, so this is safe to call here
        // even though earlier inferDepth() calls may still be in flight.
        if (d_disp_for_depth_) {
            cudaFree(d_disp_for_depth_);
            d_disp_for_depth_ = nullptr;
        }
        depth_alloc_pixels_ = 0;
        cudaMallocChecked(reinterpret_cast<void**>(&d_disp_for_depth_),
                          num_pixels * sizeof(float), "d_disp_for_depth_");
        depth_alloc_pixels_ = static_cast<int64_t>(num_pixels);
    }

    infer(d_left_rgb, d_right_rgb, input_h, input_w, d_disp_for_depth_);
    cuda::ffsCudaDispToDepth(d_disp_for_depth_, d_depth_out,
                             input_h, input_w, fx, baseline_m, stream_);
}

}  // namespace ffs_depth
