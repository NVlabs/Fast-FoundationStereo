#include "ffs_gwc_plugin.hpp"

#include <NvInfer.h>
#include <NvInferRuntimePlugin.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <iostream>
#include <mutex>
#include <string>
#include <vector>

namespace ffs_depth {
namespace cuda {
extern "C" {

void ffsCudaBuildGWCVolumeFloat(
    const float* d_fl, const float* d_fr, float* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s);

void ffsCudaBuildGWCVolumeHalf(
    const __half* d_fl, const __half* d_fr, __half* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s);

void ffsCudaBuildGWCVolumeHalfToFloat(
    const __half* d_fl, const __half* d_fr, float* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s);

void ffsCudaBuildGWCVolumeMixed(
    const float* d_fl, const float* d_fr, __half* d_gwc,
    int B, int C, int H, int W, int max_disp, int ngroups, bool normalize, cudaStream_t s);

}  // extern "C"
}  // namespace cuda

namespace {

constexpr char kPluginName[] = "FFSGWCVolume";
constexpr char kPluginVersion[] = "1";

struct GWCParams {
    int32_t max_disp = 0;   // Disparity levels at feature resolution, e.g. max_disp / 4.
    int32_t cv_group = 0;
    int32_t normalize = 1;
};

int32_t fieldToInt(nvinfer1::PluginField const& field, int32_t fallback) {
    if (!field.data || field.length <= 0) return fallback;
    if (field.type == nvinfer1::PluginFieldType::kINT32) {
        return *static_cast<int32_t const*>(field.data);
    }
    if (field.type == nvinfer1::PluginFieldType::kINT64) {
        return static_cast<int32_t>(*static_cast<int64_t const*>(field.data));
    }
    return fallback;
}

class FFSGWCVolumePlugin final : public nvinfer1::IPluginV2DynamicExt {
public:
    explicit FFSGWCVolumePlugin(GWCParams params) : params_(params) {}

    FFSGWCVolumePlugin(void const* data, size_t length) {
        if (data && length == sizeof(GWCParams)) {
            std::memcpy(&params_, data, sizeof(GWCParams));
        }
    }

    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override {
        auto* plugin = new FFSGWCVolumePlugin(params_);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    char const* getPluginType() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    int32_t getNbOutputs() const noexcept override { return 1; }

    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex,
        nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        nvinfer1::DimsExprs out{};
        if (outputIndex != 0 || nbInputs != 2 || inputs[0].nbDims != 4) {
            out.nbDims = -1;
            return out;
        }

        out.nbDims = 5;
        out.d[0] = inputs[0].d[0];  // B
        out.d[1] = exprBuilder.constant(params_.cv_group);
        out.d[2] = exprBuilder.constant(params_.max_disp);
        out.d[3] = inputs[0].d[2];  // H
        out.d[4] = inputs[0].d[3];  // W
        return out;
    }

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::PluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override {
        if (nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= 3) return false;
        if (inOut[pos].format != nvinfer1::TensorFormat::kLINEAR) return false;

        if (pos == 0) {
            return inOut[0].type == nvinfer1::DataType::kFLOAT ||
                   inOut[0].type == nvinfer1::DataType::kHALF;
        }
        if (pos == 1) {
            return inOut[1].type == inOut[0].type;
        }
        return inOut[2].type == nvinfer1::DataType::kFLOAT ||
               inOut[2].type == nvinfer1::DataType::kHALF;
    }

    void configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const*,
        int32_t,
        nvinfer1::DynamicPluginTensorDesc const*,
        int32_t) noexcept override {}

    size_t getWorkspaceSize(
        nvinfer1::PluginTensorDesc const*,
        int32_t,
        nvinfer1::PluginTensorDesc const*,
        int32_t) const noexcept override {
        return 0;
    }

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void*,
        cudaStream_t stream) noexcept override {
        if (!inputs || !outputs || !inputs[0] || !inputs[1] || !outputs[0]) return 1;
        if (inputDesc[0].dims.nbDims != 4) return 1;

        const int B = inputDesc[0].dims.d[0];
        const int C = inputDesc[0].dims.d[1];
        const int H = inputDesc[0].dims.d[2];
        const int W = inputDesc[0].dims.d[3];
        const bool normalize = params_.normalize != 0;

        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT &&
            outputDesc[0].type == nvinfer1::DataType::kFLOAT) {
            cuda::ffsCudaBuildGWCVolumeFloat(
                static_cast<float const*>(inputs[0]),
                static_cast<float const*>(inputs[1]),
                static_cast<float*>(outputs[0]),
                B, C, H, W, params_.max_disp, params_.cv_group, normalize, stream);
            return 0;
        }
        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT &&
            outputDesc[0].type == nvinfer1::DataType::kHALF) {
            cuda::ffsCudaBuildGWCVolumeMixed(
                static_cast<float const*>(inputs[0]),
                static_cast<float const*>(inputs[1]),
                static_cast<__half*>(outputs[0]),
                B, C, H, W, params_.max_disp, params_.cv_group, normalize, stream);
            return 0;
        }
        if (inputDesc[0].type == nvinfer1::DataType::kHALF &&
            outputDesc[0].type == nvinfer1::DataType::kHALF) {
            cuda::ffsCudaBuildGWCVolumeHalf(
                static_cast<__half const*>(inputs[0]),
                static_cast<__half const*>(inputs[1]),
                static_cast<__half*>(outputs[0]),
                B, C, H, W, params_.max_disp, params_.cv_group, normalize, stream);
            return 0;
        }
        if (inputDesc[0].type == nvinfer1::DataType::kHALF &&
            outputDesc[0].type == nvinfer1::DataType::kFLOAT) {
            cuda::ffsCudaBuildGWCVolumeHalfToFloat(
                static_cast<__half const*>(inputs[0]),
                static_cast<__half const*>(inputs[1]),
                static_cast<float*>(outputs[0]),
                B, C, H, W, params_.max_disp, params_.cv_group, normalize, stream);
            return 0;
        }

        return 1;
    }

    nvinfer1::DataType getOutputDataType(
        int32_t,
        nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override {
        if (nbInputs > 0 && inputTypes[0] == nvinfer1::DataType::kHALF) {
            return nvinfer1::DataType::kHALF;
        }
        return nvinfer1::DataType::kFLOAT;
    }

    int32_t initialize() noexcept override { return 0; }
    void terminate() noexcept override {}
    size_t getSerializationSize() const noexcept override { return sizeof(GWCParams); }
    void serialize(void* buffer) const noexcept override {
        std::memcpy(buffer, &params_, sizeof(GWCParams));
    }
    void destroy() noexcept override { delete this; }
    void setPluginNamespace(char const* pluginNamespace) noexcept override {
        namespace_ = pluginNamespace ? pluginNamespace : "";
    }
    char const* getPluginNamespace() const noexcept override { return namespace_.c_str(); }

    void attachToContext(cudnnContext*, cublasContext*, nvinfer1::IGpuAllocator*) noexcept override {}
    void detachFromContext() noexcept override {}

private:
    GWCParams params_;
    std::string namespace_;
};

class FFSGWCVolumePluginCreator final : public nvinfer1::IPluginCreator {
public:
    FFSGWCVolumePluginCreator() {
        fields_.emplace_back("max_disp", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        fields_.emplace_back("cv_group", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        fields_.emplace_back("normalize", nullptr, nvinfer1::PluginFieldType::kINT32, 1);
        field_collection_.nbFields = static_cast<int32_t>(fields_.size());
        field_collection_.fields = fields_.data();
    }

    char const* getPluginName() const noexcept override { return kPluginName; }
    char const* getPluginVersion() const noexcept override { return kPluginVersion; }
    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        return &field_collection_;
    }

    nvinfer1::IPluginV2* createPlugin(
        char const*,
        nvinfer1::PluginFieldCollection const* fc) noexcept override {
        GWCParams params;
        if (fc) {
            for (int32_t i = 0; i < fc->nbFields; ++i) {
                auto const& field = fc->fields[i];
                if (!std::strcmp(field.name, "max_disp")) {
                    params.max_disp = fieldToInt(field, params.max_disp);
                } else if (!std::strcmp(field.name, "cv_group")) {
                    params.cv_group = fieldToInt(field, params.cv_group);
                } else if (!std::strcmp(field.name, "normalize")) {
                    params.normalize = fieldToInt(field, params.normalize);
                }
            }
        }
        if (params.max_disp <= 0 || params.cv_group <= 0) return nullptr;
        auto* plugin = new FFSGWCVolumePlugin(params);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    nvinfer1::IPluginV2* deserializePlugin(
        char const*,
        void const* serialData,
        size_t serialLength) noexcept override {
        if (serialLength != sizeof(GWCParams)) return nullptr;
        auto* plugin = new FFSGWCVolumePlugin(serialData, serialLength);
        plugin->setPluginNamespace(namespace_.c_str());
        return plugin;
    }

    void setPluginNamespace(char const* pluginNamespace) noexcept override {
        namespace_ = pluginNamespace ? pluginNamespace : "";
    }
    char const* getPluginNamespace() const noexcept override {
        return namespace_.c_str();
    }

private:
    std::vector<nvinfer1::PluginField> fields_;
    nvinfer1::PluginFieldCollection field_collection_{};
    std::string namespace_;
};

FFSGWCVolumePluginCreator g_creator;

}  // namespace

bool registerFFSGWCPlugin() {
    static bool result = false;
    static std::once_flag once;
    std::call_once(once, []() {
        auto try_register = [](nvinfer1::IPluginRegistry* registry, const char* tag) -> bool {
            if (!registry) {
                std::cerr << "[FFS plugin] " << tag << ": registry unavailable\n";
                return false;
            }
            if (registry->getPluginCreator(kPluginName, kPluginVersion, "")) {
                std::cerr << "[FFS plugin] " << tag
                          << ": FFSGWCVolume v1 ns=\"\" found-already\n";
                return true;
            }
            if (registry->registerCreator(g_creator, "")) {
                std::cerr << "[FFS plugin] " << tag
                          << ": FFSGWCVolume v1 ns=\"\" newly-registered\n";
                return true;
            }
            std::cerr << "[FFS plugin] " << tag
                      << ": FFSGWCVolume v1 ns=\"\" register FAILED\n";
            return false;
        };

        const bool a = try_register(::getPluginRegistry(), "runtime-registry");
        const bool b = try_register(
            nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD),
            "builder-registry");
        result = a || b;
    });
    return result;
}

// Auto-register the creator with the global TRT plugin registry on shared
// library load. Lets external tools (trtexec --staticPlugins, polygraphy
// --plugins) deserialize engines containing FFSGWCVolume without calling
// registerFFSGWCPlugin() themselves.
REGISTER_TENSORRT_PLUGIN(FFSGWCVolumePluginCreator);

}  // namespace ffs_depth

extern "C" bool ffs_register_gwc_plugin() {
    return ffs_depth::registerFFSGWCPlugin();
}
