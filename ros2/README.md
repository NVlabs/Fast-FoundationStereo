# ROS 2 Stereo Depth Pipeline (ZED camera → Fast-FoundationStereo → depth topic)

Dockerized ROS 2 (Humble) inference pipeline built on the [C++ TensorRT runtime](../cpp/README.md):
the `ffs_stereo_node` subscribes to the rectified left/right images of a ZED camera (published by
the official [zed-ros2-wrapper](https://github.com/stereolabs/zed-ros2-wrapper)), runs
Fast-FoundationStereo on the GPU, and publishes metric depth as a `sensor_msgs/Image`.

```text
┌──────────────────────┐  /zed/zed_node/left/image_rect_color    ┌─────────────────────────────┐
│   zed-ros2-wrapper   │  /zed/zed_node/right/image_rect_color   │   ffs_stereo_node (this)    │
│  (ZED SDK capture +  │ ───────────────────────────────────────▶│  C++ TensorRT runtime from  │
│   rectification)     │  /zed/zed_node/{left,right}/camera_info │  cpp/ (GPU preprocessing,   │
└──────────────────────┘                                         │  inference, disp→depth)     │
   host or its own container                                     └──────────────┬──────────────┘
                                                                                │
                                                  /ffs_stereo/depth/image_rect (32FC1, meters) +
                                                  /ffs_stereo/depth/camera_info
                                                  [optional] /ffs_stereo/disparity/image
```

Why subscribe to wrapper topics instead of linking the ZED SDK directly?

- The wrapper already handles capture, rectification, timestamps, and calibration publishing;
  `fx` and the stereo baseline are derived automatically from the two `camera_info` topics
  (`baseline = -P[3] / P[0]` of the right camera), so no `K.txt` file is needed.
- This image stays free of the ZED SDK (no SDK/CUDA version coupling, works with rosbags too).
- You can disable the ZED SDK's own depth computation (`depth_mode: NONE` in the wrapper config)
  so the GPU is spent on Fast-FoundationStereo instead.

The node auto-detects the engine layout the same way `cpp/build/ffs_depth_main` does:
`fast_foundationstereo.engine` present → single-engine plugin route, otherwise
`feature_runner.engine` + `post_runner.engine` → two-engine route.

## 1. Build the docker images (host with NVIDIA driver + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html))

```bash
# Python/PyTorch image: only needed once, to export the ONNX
docker build --network host -t ffs -f docker/dockerfile_cpp .

# ROS 2 runtime image: builds cpp/ + the ffs_ros workspace
docker build --network host -t ffs_ros2 -f docker/dockerfile_ros2 .
```

For GPUs outside the default SM list (Ampere–Hopper), pass the architecture, e.g.
`--build-arg FFS_CUDA_ARCHS=87` for Jetson Orin (Jetson also needs an L4T/arm64 base image —
the dockerfiles here target x86_64).

## 2. Export the plugin ONNX (once, any machine)

Download the weights as described in the [top-level README](../readme.md#weights-and-trade-off),
then export from inside the `ffs` container (`bash docker/run_container.sh`):

```bash
python3 scripts/make_plugin_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output_plugin_onnx \
  --height 480 --width 640 \
  --valid_iters 8 --max_disp 192
```

This writes `output_plugin_onnx/fast_foundationstereo_plugin.onnx` + `onnx.yaml`. Pick
`--height/--width` ≤ your camera resolution (divisible by 32); the runtime aspect-resizes any
input to the model size and returns depth at full camera resolution. Smaller model size and
`--valid_iters 4` → faster.

## 3. Build the TensorRT engine (on the deployment machine)

Engines are specific to the GPU **and** the TensorRT version, so build inside the `ffs_ros2`
image on the machine that will run inference:

```bash
ENGINE_DIR=$(pwd)/output_plugin_onnx bash docker/run_ros2_container.sh \
  ros2 run ffs_ros ffs_build_single_engine \
  /engines/fast_foundationstereo_plugin.onnx /engines/fast_foundationstereo.engine
```

`/engines` now holds `fast_foundationstereo.engine` + `onnx.yaml` — everything the node needs.
(Two-engine alternative: export with `scripts/make_onnx.py` and build both engines with
`trtexec --fp16`, which is available in the image at `/usr/src/tensorrt/bin/trtexec`.)

## 4. Run the ZED wrapper

Run the official wrapper wherever you prefer — natively on the host:

```bash
ros2 launch zed_wrapper zed_camera.launch.py camera_model:=zed2i   # or zed, zedx, zed2, zedm...
```

or in Stereolabs' own container (see the
[zed-ros2-wrapper docker instructions](https://github.com/stereolabs/zed-ros2-wrapper/tree/master/docker);
it needs `--gpus all --privileged -v /dev:/dev` for USB access).

Recommended wrapper settings:

- Set the wrapper's depth mode to `NONE` (`depth.depth_mode` in `common_stereo.yaml` or via a
  params override) — Fast-FoundationStereo replaces the SDK depth, no point computing it twice.
- Keep the publish resolution moderate (e.g. HD720); the model resizes internally anyway and
  smaller images reduce transport and copy overhead.

## 5. Run the inference node

```bash
ENGINE_DIR=$(pwd)/output_plugin_onnx bash docker/run_ros2_container.sh \
  ros2 launch ffs_ros ffs_stereo.launch.py engine_dir:=/engines camera_name:=zed
```

`camera_name` must match the wrapper's camera name so the topic names line up
(`/<camera_name>/zed_node/left/image_rect_color`, etc.). For non-ZED topic layouts, set the
`*_topic` parameters directly (see `ros2/ffs_ros/config/ffs_stereo.yaml`).

First inference takes a few seconds (engine warm-up); after that you should see depth at
camera rate, e.g.:

```bash
ros2 topic hz /ffs_stereo/depth/image_rect
ros2 topic echo --no-arr /ffs_stereo/depth/image_rect   # encoding: 32FC1, meters
```

To get a point cloud, feed the depth + camera_info into `depth_image_proc` (preinstalled in
the image):

```bash
ros2 run depth_image_proc point_cloud_xyz_node --ros-args \
  -r image_rect:=/ffs_stereo/depth/image_rect \
  -r camera_info:=/ffs_stereo/depth/camera_info
```

## Node reference

`ffs_stereo_node` parameters:

| Parameter | Default | Meaning |
|---|---|---|
| `engine_dir` | `""` (required) | Directory with the engine(s) + `onnx.yaml` |
| `left_image_topic` / `right_image_topic` | `/zed/zed_node/{left,right}/image_rect_color` | Rectified stereo pair (any cv_bridge-convertible encoding; ZED's `bgra8` works) |
| `left_camera_info_topic` / `right_camera_info_topic` | `/zed/zed_node/{left,right}/camera_info` | Source of `fx` and baseline |
| `depth_topic` / `depth_camera_info_topic` | `~/depth/image_rect`, `~/depth/camera_info` | Outputs (`~/` = under the node name) |
| `publish_disparity` / `disparity_topic` | `false`, `~/disparity/image` | Optional raw disparity (32FC1, input-pixel units) |
| `fx`, `baseline_m` | `0.0` | Manual calibration override; `0.0` = derive from camera_info |
| `max_depth` | `0.0` | Depth beyond this (meters) becomes NaN; `0.0` disables |
| `sync_queue_size` | `10` | ApproximateTime queue for pairing left/right |

Behavior notes:

- Depth is published in the **left rectified frame** (the header/frame_id of the incoming left
  image is reused), so TF from the ZED wrapper applies unchanged.
- Invalid pixels (zero disparity, correspondence off-image) are **NaN** per REP 118; consumers
  like `depth_image_proc`, Nav2 and RTAB-Map skip NaNs.
- Image subscriptions use best-effort sensor QoS (compatible with both reliable and
  best-effort publishers); depth is published reliable.
- Inference runs synchronously in the subscription callback; if the camera outpaces the model,
  old pairs are dropped from the sync queue — you always get the freshest depth.

## Troubleshooting

- **No depth output, node logs "Waiting for stereo pairs"** — check topic names
  (`ros2 topic list`), `ROS_DOMAIN_ID`, and that both containers/host share the network
  (`--network host`).
- **"Waiting for camera_info"** — the wrapper publishes camera_info alongside the images;
  verify the `*_camera_info_topic` parameters, or set `fx` and `baseline_m` manually.
- **Engine fails to load / version error** — the engine must be built by the same TensorRT
  version that runs it: rebuild it inside `ffs_ros2` (step 3), don't copy engines across
  machines or images.
- **Slow / GPU-starved** — set the ZED wrapper `depth_mode` to `NONE`, export a smaller model
  size (e.g. 320×512), or reduce `--valid_iters` to 4 at export time.

## ROS 1?

The node is ~300 lines of plain `rclcpp` + `message_filters` + `cv_bridge`; the same structure
maps 1:1 onto ROS 1 Noetic (`ros::NodeHandle`, identical message_filters API) if you need it —
the C++ runtime in `cpp/` is ROS-agnostic.
