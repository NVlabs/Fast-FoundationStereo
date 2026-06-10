#!/usr/bin/env bash
# Run the ffs_ros2 image with GPU access and host networking (so DDS can
# discover the ZED wrapper running on the host or in another container).
#
# ENGINE_DIR (default ./output_plugin_onnx) is mounted at /engines and must
# contain the TensorRT engine(s) + onnx.yaml.
#
# Examples:
#   bash docker/run_ros2_container.sh                                   # interactive shell
#   bash docker/run_ros2_container.sh ros2 launch ffs_ros ffs_stereo.launch.py engine_dir:=/engines camera_name:=zed
set -e

ENGINE_DIR=${ENGINE_DIR:-$(pwd)/output_plugin_onnx}
mkdir -p "$ENGINE_DIR"

docker rm -f ffs_ros2 2>/dev/null || true
docker run --gpus all -it --rm \
  --name ffs_ros2 \
  --network host \
  --ipc host \
  -e NVIDIA_DISABLE_REQUIRE=1 \
  -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}" \
  -v "$(realpath "$ENGINE_DIR")":/engines \
  ffs_ros2 "$@"
