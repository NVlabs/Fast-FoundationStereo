/**
 * ROS 2 node wrapping the Fast-FoundationStereo TensorRT C++ runtime (cpp/).
 *
 * Subscribes to time-synchronized rectified left/right images (e.g. from the
 * zed-ros2-wrapper) plus both camera_info topics, runs stereo inference on
 * the GPU, and publishes:
 *   - ~/depth/image_rect   sensor_msgs/Image, 32FC1, meters, aligned to the
 *                          left rectified frame. Invalid pixels are NaN.
 *   - ~/depth/camera_info  the left CameraInfo restamped with the image header
 *   - ~/disparity/image    optional, 32FC1, input-pixel units
 *
 * fx is taken from the left CameraInfo projection matrix (P[0]) and the
 * baseline from the right CameraInfo (baseline = -P[3] / P[0]); both can be
 * overridden with the `fx` / `baseline_m` parameters.
 *
 * The engine directory is the same layout the cpp/ apps use:
 *   - fast_foundationstereo.engine + onnx.yaml  -> single-engine plugin route
 *   - feature_runner.engine + post_runner.engine + onnx.yaml -> two-engine route
 */

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <utility>

#include <cv_bridge/cv_bridge.h>
#include <message_filters/subscriber.h>
#include <message_filters/sync_policies/approximate_time.h>
#include <message_filters/synchronizer.h>
#include <opencv2/core.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/image_encodings.hpp>
#include <sensor_msgs/msg/camera_info.hpp>
#include <sensor_msgs/msg/image.hpp>

#include "ffs_depth_single_tensorrt.hpp"
#include "ffs_depth_tensorrt.hpp"

namespace {

void checkCuda(cudaError_t status, const char* what) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(status));
    }
}

void sanitizeDepth(float* depth, size_t count, float max_depth) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    for (size_t i = 0; i < count; ++i) {
        const float d = depth[i];
        if (!std::isfinite(d) || d <= 0.0f || (max_depth > 0.0f && d > max_depth)) {
            depth[i] = nan;
        }
    }
}

}  // namespace

class FFSStereoNode : public rclcpp::Node {
public:
    FFSStereoNode() : rclcpp::Node("ffs_stereo") {
        engine_dir_ = declare_parameter<std::string>("engine_dir", "");
        const auto left_image_topic =
            declare_parameter<std::string>("left_image_topic", "/zed/zed_node/left/image_rect_color");
        const auto right_image_topic =
            declare_parameter<std::string>("right_image_topic", "/zed/zed_node/right/image_rect_color");
        const auto left_info_topic =
            declare_parameter<std::string>("left_camera_info_topic", "/zed/zed_node/left/camera_info");
        const auto right_info_topic =
            declare_parameter<std::string>("right_camera_info_topic", "/zed/zed_node/right/camera_info");
        const auto depth_topic = declare_parameter<std::string>("depth_topic", "~/depth/image_rect");
        const auto depth_info_topic =
            declare_parameter<std::string>("depth_camera_info_topic", "~/depth/camera_info");
        const auto disparity_topic =
            declare_parameter<std::string>("disparity_topic", "~/disparity/image");
        publish_disparity_ = declare_parameter<bool>("publish_disparity", false);
        fx_param_ = static_cast<float>(declare_parameter<double>("fx", 0.0));
        baseline_param_ = static_cast<float>(declare_parameter<double>("baseline_m", 0.0));
        max_depth_ = static_cast<float>(declare_parameter<double>("max_depth", 0.0));
        const int sync_queue = static_cast<int>(declare_parameter<int>("sync_queue_size", 10));

        if (engine_dir_.empty()) {
            throw std::runtime_error(
                "Parameter 'engine_dir' is required. It must point to a directory containing "
                "either fast_foundationstereo.engine + onnx.yaml (single-engine route) or "
                "feature_runner.engine + post_runner.engine + onnx.yaml (two-engine route).");
        }

        const bool use_single_engine =
            std::filesystem::exists(std::filesystem::path(engine_dir_) / "fast_foundationstereo.engine");
        RCLCPP_INFO(get_logger(), "Loading TensorRT engine(s) from %s (%s route)",
                    engine_dir_.c_str(), use_single_engine ? "single-engine plugin" : "two-engine");
        if (use_single_engine) {
            single_ = std::make_unique<ffs_depth::FFSSingleEngineInference>(engine_dir_);
        } else {
            two_ = std::make_unique<ffs_depth::FFSDepthInference>(engine_dir_);
        }
        withEngine([&](auto& eng) {
            RCLCPP_INFO(get_logger(), "Model input %dx%d, max_disp=%d, valid_iters=%d",
                        eng.modelWidth(), eng.modelHeight(), eng.maxDisp(), eng.config().valid_iters);
        });

        depth_pub_ = create_publisher<sensor_msgs::msg::Image>(depth_topic, rclcpp::QoS(5));
        depth_info_pub_ = create_publisher<sensor_msgs::msg::CameraInfo>(depth_info_topic, rclcpp::QoS(5));
        if (publish_disparity_) {
            disparity_pub_ = create_publisher<sensor_msgs::msg::Image>(disparity_topic, rclcpp::QoS(5));
        }

        // Best-effort subscriptions are QoS-compatible with both reliable and
        // best-effort publishers, so they work with any zed-ros2-wrapper config.
        left_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            left_info_topic, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
                std::lock_guard<std::mutex> lock(info_mutex_);
                left_info_ = std::move(msg);
            });
        right_info_sub_ = create_subscription<sensor_msgs::msg::CameraInfo>(
            right_info_topic, rclcpp::SensorDataQoS(),
            [this](sensor_msgs::msg::CameraInfo::ConstSharedPtr msg) {
                std::lock_guard<std::mutex> lock(info_mutex_);
                right_info_ = std::move(msg);
            });

        left_image_sub_.subscribe(this, left_image_topic, rmw_qos_profile_sensor_data);
        right_image_sub_.subscribe(this, right_image_topic, rmw_qos_profile_sensor_data);
        sync_ = std::make_shared<message_filters::Synchronizer<SyncPolicy>>(
            SyncPolicy(sync_queue), left_image_sub_, right_image_sub_);
        sync_->registerCallback(
            std::bind(&FFSStereoNode::imagesCallback, this, std::placeholders::_1, std::placeholders::_2));

        RCLCPP_INFO(get_logger(), "Waiting for stereo pairs on %s + %s",
                    left_image_topic.c_str(), right_image_topic.c_str());
    }

    ~FFSStereoNode() override { freeBuffers(); }

private:
    using SyncPolicy = message_filters::sync_policies::ApproximateTime<sensor_msgs::msg::Image,
                                                                       sensor_msgs::msg::Image>;

    template <typename F>
    void withEngine(F&& f) {
        if (single_) {
            f(*single_);
        } else {
            f(*two_);
        }
    }

    void freeBuffers() {
        if (d_left_) cudaFree(d_left_);
        if (d_right_) cudaFree(d_right_);
        if (d_disp_) cudaFree(d_disp_);
        if (d_depth_) cudaFree(d_depth_);
        d_left_ = d_right_ = nullptr;
        d_disp_ = d_depth_ = nullptr;
        buf_h_ = buf_w_ = 0;
    }

    void ensureBuffers(int h, int w) {
        if (h == buf_h_ && w == buf_w_) return;
        freeBuffers();
        const size_t image_bytes = static_cast<size_t>(h) * w * 3;
        const size_t map_bytes = static_cast<size_t>(h) * w * sizeof(float);
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_left_), image_bytes), "cudaMalloc left");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_right_), image_bytes), "cudaMalloc right");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_disp_), map_bytes), "cudaMalloc disparity");
        checkCuda(cudaMalloc(reinterpret_cast<void**>(&d_depth_), map_bytes), "cudaMalloc depth");
        buf_h_ = h;
        buf_w_ = w;
    }

    bool lookupCalibration(float& fx, float& baseline,
                           sensor_msgs::msg::CameraInfo::ConstSharedPtr& left_info) {
        sensor_msgs::msg::CameraInfo::ConstSharedPtr li;
        sensor_msgs::msg::CameraInfo::ConstSharedPtr ri;
        {
            std::lock_guard<std::mutex> lock(info_mutex_);
            li = left_info_;
            ri = right_info_;
        }
        left_info = li;

        fx = fx_param_;
        if (fx <= 0.0f && li) {
            // Rectified streams: use the projection matrix focal length.
            fx = static_cast<float>(li->p[0] > 0.0 ? li->p[0] : li->k[0]);
        }

        baseline = baseline_param_;
        if (baseline <= 0.0f && ri) {
            const double fx_right = ri->p[0] > 0.0 ? ri->p[0] : ri->k[0];
            if (fx_right > 0.0) {
                // Right CameraInfo of a rectified pair has P[3] = -fx * baseline.
                baseline = static_cast<float>(std::abs(ri->p[3] / fx_right));
            }
        }

        return fx > 0.0f && baseline > 0.0f;
    }

    void imagesCallback(const sensor_msgs::msg::Image::ConstSharedPtr& left,
                        const sensor_msgs::msg::Image::ConstSharedPtr& right) {
        try {
            processPair(left, right);
        } catch (const std::exception& e) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "Inference failed: %s", e.what());
        }
    }

    void processPair(const sensor_msgs::msg::Image::ConstSharedPtr& left,
                     const sensor_msgs::msg::Image::ConstSharedPtr& right) {
        float fx = 0.0f;
        float baseline = 0.0f;
        sensor_msgs::msg::CameraInfo::ConstSharedPtr left_info;
        if (!lookupCalibration(fx, baseline, left_info)) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "Waiting for camera_info (or set the 'fx' and 'baseline_m' "
                                 "parameters); dropping stereo pair");
            return;
        }

        cv_bridge::CvImageConstPtr left_cv;
        cv_bridge::CvImageConstPtr right_cv;
        try {
            left_cv = cv_bridge::toCvCopy(left, sensor_msgs::image_encodings::BGR8);
            right_cv = cv_bridge::toCvCopy(right, sensor_msgs::image_encodings::BGR8);
        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR_THROTTLE(get_logger(), *get_clock(), 5000, "cv_bridge conversion failed: %s",
                                  e.what());
            return;
        }
        if (left_cv->image.size() != right_cv->image.size()) {
            RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                                 "Left (%dx%d) and right (%dx%d) image sizes differ; dropping pair",
                                 left_cv->image.cols, left_cv->image.rows, right_cv->image.cols,
                                 right_cv->image.rows);
            return;
        }

        // The GPU upload assumes packed rows.
        cv::Mat left_bgr = left_cv->image;
        cv::Mat right_bgr = right_cv->image;
        if (!left_bgr.isContinuous()) left_bgr = left_bgr.clone();
        if (!right_bgr.isContinuous()) right_bgr = right_bgr.clone();

        const int h = left_bgr.rows;
        const int w = left_bgr.cols;
        const size_t image_bytes = static_cast<size_t>(h) * w * 3;
        const size_t map_bytes = static_cast<size_t>(h) * w * sizeof(float);
        ensureBuffers(h, w);

        const auto t0 = std::chrono::steady_clock::now();
        checkCuda(cudaMemcpy(d_left_, left_bgr.data, image_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy left image");
        checkCuda(cudaMemcpy(d_right_, right_bgr.data, image_bytes, cudaMemcpyHostToDevice),
                  "cudaMemcpy right image");

        withEngine([&](auto& eng) {
            if (publish_disparity_) {
                eng.infer(d_left_, d_right_, h, w, d_disp_);
                eng.dispToDepth(d_disp_, h, w, fx, baseline, d_depth_);
            } else {
                eng.inferDepth(d_left_, d_right_, h, w, fx, baseline, d_depth_);
            }
            eng.sync();
        });

        auto depth_msg = std::make_unique<sensor_msgs::msg::Image>();
        depth_msg->header = left->header;
        depth_msg->height = static_cast<uint32_t>(h);
        depth_msg->width = static_cast<uint32_t>(w);
        depth_msg->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
        depth_msg->is_bigendian = 0;
        depth_msg->step = static_cast<uint32_t>(w * sizeof(float));
        depth_msg->data.resize(map_bytes);
        checkCuda(cudaMemcpy(depth_msg->data.data(), d_depth_, map_bytes, cudaMemcpyDeviceToHost),
                  "cudaMemcpy depth");
        sanitizeDepth(reinterpret_cast<float*>(depth_msg->data.data()),
                      static_cast<size_t>(h) * w, max_depth_);
        depth_pub_->publish(std::move(depth_msg));

        if (left_info) {
            auto info_msg = std::make_unique<sensor_msgs::msg::CameraInfo>(*left_info);
            info_msg->header = left->header;
            depth_info_pub_->publish(std::move(info_msg));
        }

        if (publish_disparity_ && disparity_pub_) {
            auto disp_msg = std::make_unique<sensor_msgs::msg::Image>();
            disp_msg->header = left->header;
            disp_msg->height = static_cast<uint32_t>(h);
            disp_msg->width = static_cast<uint32_t>(w);
            disp_msg->encoding = sensor_msgs::image_encodings::TYPE_32FC1;
            disp_msg->is_bigendian = 0;
            disp_msg->step = static_cast<uint32_t>(w * sizeof(float));
            disp_msg->data.resize(map_bytes);
            checkCuda(cudaMemcpy(disp_msg->data.data(), d_disp_, map_bytes, cudaMemcpyDeviceToHost),
                      "cudaMemcpy disparity");
            disparity_pub_->publish(std::move(disp_msg));
        }

        const double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        if (frame_count_++ == 0) {
            RCLCPP_INFO(get_logger(),
                        "Publishing depth: %dx%d, fx=%.2f, baseline=%.4f m, first frame %.1f ms",
                        w, h, fx, baseline, ms);
        } else {
            RCLCPP_DEBUG(get_logger(), "Stereo pair processed in %.1f ms", ms);
        }
    }

    std::string engine_dir_;
    bool publish_disparity_ = false;
    float fx_param_ = 0.0f;
    float baseline_param_ = 0.0f;
    float max_depth_ = 0.0f;

    std::unique_ptr<ffs_depth::FFSSingleEngineInference> single_;
    std::unique_ptr<ffs_depth::FFSDepthInference> two_;

    uint8_t* d_left_ = nullptr;
    uint8_t* d_right_ = nullptr;
    float* d_disp_ = nullptr;
    float* d_depth_ = nullptr;
    int buf_h_ = 0;
    int buf_w_ = 0;
    uint64_t frame_count_ = 0;

    std::mutex info_mutex_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr left_info_;
    sensor_msgs::msg::CameraInfo::ConstSharedPtr right_info_;

    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr depth_pub_;
    rclcpp::Publisher<sensor_msgs::msg::CameraInfo>::SharedPtr depth_info_pub_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr disparity_pub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr left_info_sub_;
    rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr right_info_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> left_image_sub_;
    message_filters::Subscriber<sensor_msgs::msg::Image> right_image_sub_;
    std::shared_ptr<message_filters::Synchronizer<SyncPolicy>> sync_;
};

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    int ret = 0;
    try {
        rclcpp::spin(std::make_shared<FFSStereoNode>());
    } catch (const std::exception& e) {
        RCLCPP_FATAL(rclcpp::get_logger("ffs_stereo"), "%s", e.what());
        ret = 1;
    }
    rclcpp::shutdown();
    return ret;
}
