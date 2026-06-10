# Fast-FoundationStereo C++ Inference

C++ runtime for Fast-FoundationStereo stereo depth inference on TensorRT.

For a dockerized ROS 2 node that feeds this runtime from a live ZED camera and publishes depth as a ROS topic, see [`ros2/README.md`](../ros2/README.md).

## Run from inside the C++ Docker container

**All commands in this README are intended to be run from inside the container built from [`docker/dockerfile_cpp`](../docker/dockerfile_cpp).** That image carries the CUDA toolkit, TensorRT 10 runtime + ONNX parser + C++ headers, OpenCV development package, `trtexec`, and a Python environment with PyTorch / ONNX export tooling. The plain `docker/dockerfile` image does **not** ship the C++ TRT headers and will fail to build the C++ targets.

Environment setup:

```bash
docker build --network host -t ffs -f docker/dockerfile_cpp .
bash docker/run_container.sh
```


`run_container.sh` mounts the parent of `docker/` (the repo root) into `/workspace`, so `cd /workspace/<repo-name>/cpp` lands you in this folder. Every `cd cpp`, `cpp/build/...`, `python3 scripts/...`, and `trtexec ...` example below assumes that working directory.

```text
cpp/
+-- CMakeLists.txt
+-- README.md
+-- app/
|   +-- main.cpp                  -- ffs_depth_main: single image inference + visualization
|   +-- build_single_engine.cpp   -- ffs_build_single_engine: ONNX (with FFSGWCVolume plugin) -> .engine
|   +-- profile_speed.cpp         -- ffs_profile_speed: latency profiler for either route
+-- include/
|   +-- ffs_depth_tensorrt.hpp
|   +-- ffs_depth_single_tensorrt.hpp
|   +-- ffs_gwc_plugin.hpp
+-- src/
    +-- ffs_depth_tensorrt.cpp
    +-- ffs_depth_single_tensorrt.cpp
    +-- gwc_volume_plugin.cpp
    +-- depth_kernels.cu
```

Two interchangeable inference routes are supported:

- `FFSSingleEngineInference` loads **one** TensorRT engine that contains the `FFSGWCVolume` plugin node.
- `FFSDepthInference` loads **two** TensorRT engines (feature_runner + post_runner) and computes the GWC cost volume between them with a hand-written CUDA kernel.

`ffs_depth_main` auto-detects the route based on the engine directory contents:

- If the directory contains `fast_foundationstereo.engine`, it uses the single-engine plugin path.
- Otherwise it uses the two-engine reference path.

Inputs and outputs are the same in both cases:

- Input: a stereo image pair plus an intrinsic file (`demo_data/K.txt` format: 9 floats on line 1 for the 3x3 camera matrix, one float on line 2 for the stereo baseline in meters).
- Output: float32 disparity (input-pixel units), float32 depth in meters, and PNG visualizations.

## Dependencies

All C++ build dependencies are provided by the [`docker/dockerfile_cpp`](../docker/dockerfile_cpp) image (see [Run from inside the C++ Docker container](#run-from-inside-the-c-docker-container) for how to build and enter it):

- CUDA Toolkit (for `nvcc` and the CUDA runtime).
- TensorRT 10 runtime, ONNX parser, and C++ headers.
- OpenCV development package (image I/O and depth visualization in `app/main.cpp`).
- `trtexec`, used by Route B to build the two-engine TensorRT engines from ONNX.

## Build

```bash
cd cpp
cmake -B build
cmake --build build -j
```

This produces:

- `build/libffs_gwc_plugin.so` -- the FFSGWCVolume plugin as a shared library (loadable by `trtexec --staticPlugins=...`).
- `build/libffs_depth_inference.a` -- the inference static library.
- `build/ffs_build_single_engine` -- single-engine builder.
- `build/ffs_depth_main` -- demo / inference CLI.
- `build/ffs_profile_speed` -- latency profiler.

## Route A: Single Engine with FFSGWCVolume Plugin

### A.1 Export ONNX with the plugin node

`scripts/make_plugin_onnx.py` exports one ONNX graph in which the GWC cost volume is represented by an `FFSGWCVolume` custom plugin node (resolved at engine-build time by `libffs_gwc_plugin.so`):

```bash
python3 scripts/make_plugin_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output_plugin_onnx \
  --height 480 \
  --width 640 \
  --valid_iters 8 \
  --max_disp 192
```

This writes:

```text
output_plugin_onnx/
+-- fast_foundationstereo_plugin.onnx
+-- onnx.yaml
```

### A.2 Build the single TensorRT engine

```bash
cpp/build/ffs_build_single_engine \
  output_plugin_onnx/fast_foundationstereo_plugin.onnx \
  output_plugin_onnx/fast_foundationstereo.engine
```

By default the engine is built with FP16 enabled. Pass `--fp32` to disable FP16. Pass `--workspace-mb N` to override the workspace (default 4096 MB).

After this step `output_plugin_onnx/` contains everything the runtime needs:

```text
output_plugin_onnx/
+-- fast_foundationstereo.engine
+-- onnx.yaml
```

### A.3 Run inference

Run from the repository root:

```bash
cpp/build/ffs_depth_main \
  output_plugin_onnx \
  demo_data/left.png \
  demo_data/right.png \
  demo_data/K.txt \
  output_plugin_onnx
```

The last argument is the output directory (default `ffs_output`). `ffs_depth_main` sees `fast_foundationstereo.engine` inside `output_plugin_onnx/` and uses `FFSSingleEngineInference`, which deserializes the engine and registers the `FFSGWCVolume` plugin before inference.

### A.4 Python alternative (build engine and run inference without the C++ apps)

The same plugin ONNX can be turned into an engine and executed end-to-end from Python. Only the C++ plugin shared library (`libffs_gwc_plugin.so`) is required from the C++ build; the C++ apps (`ffs_build_single_engine`, `ffs_depth_main`) are not.

```bash
# 1. Export plugin ONNX
python3 scripts/make_plugin_onnx.py \
  --model_dir weights/23-36-37/model_best_bp2_serialize.pth \
  --save_path output_plugin_onnx \
  --height 480 \
  --width 640

# 2. Build the C++ plugin shared library
cmake -S cpp -B cpp/build
cmake --build cpp/build -j

# 3. Build the TensorRT engine from Python
python3 scripts/build_plugin_trt.py \
  output_plugin_onnx/fast_foundationstereo_plugin.onnx \
  output_plugin_onnx/fast_foundationstereo.engine

# 4. Run inference from Python
python3 scripts/run_demo_plugin_trt.py \
  --model_dir output_plugin_onnx \
  --left_file demo_data/left.png \
  --right_file demo_data/right.png \
  --intrinsic_file demo_data/K.txt \
  --out_dir output_plugin_onnx
```

Both `build_plugin_trt.py` and `run_demo_plugin_trt.py` auto-discover `libffs_gwc_plugin.so` in `cpp/build/`. Pass `--plugin_lib /path/to/libffs_gwc_plugin.so` to override the location. `build_plugin_trt.py` accepts `--fp32` and `--workspace-mb N` with the same meaning as `ffs_build_single_engine`. `run_demo_plugin_trt.py` writes the same five output files as the C++ demo (see [Outputs](#outputs)).

## Route B: Two TensorRT Engines (feature_runner + post_runner)

### B.1 Build the two engines

See the [Two-stage ONNX section in the top-level README](../readme.md#two-stage-onnx) for how to export `feature_runner.onnx` / `post_runner.onnx` with `scripts/make_onnx.py` and then build `feature_runner.engine` / `post_runner.engine` with `trtexec`. After running those steps your engine directory should contain:

```text
output_two_onnx/
+-- feature_runner.engine
+-- post_runner.engine
+-- onnx.yaml
```

The two engines do **not** use the FFSGWCVolume plugin: the GWC cost volume is computed externally on GPU by `cpp/src/depth_kernels.cu` between the two engine calls, so a plain `trtexec --fp16` build with no custom plugin library is enough.

### B.2 Run inference

```bash
cpp/build/ffs_depth_main \
  output_two_onnx \
  demo_data/left.png \
  demo_data/right.png \
  demo_data/K.txt \
  output_two_onnx
```

`ffs_depth_main` sees no `fast_foundationstereo.engine` in the directory and falls back to `FFSDepthInference`, which executes the feature engine, builds the GWC volume with the CUDA kernel, then executes the post engine.

## Outputs

Both routes write the same files into the output directory:

- `disparity.bin` -- raw float32 disparity (input-pixel units), prefixed by int32 `[height, width]`.
- `depth_meter.bin` -- raw float32 depth in meters, prefixed by int32 `[height, width]`.
- `depth_meter.npy` -- NumPy float32 depth in meters, shape `[height, width]`.
- `disp_vis.png` -- left/right/colorized-disparity side-by-side visualization.
- `depth_vis.png` -- colorized depth visualization.

## Profile Speed

`ffs_profile_speed` benchmarks either route end-to-end on a single image pair using CUDA events for GPU-side timing and `steady_clock` for host wall time. It auto-detects the route the same way `ffs_depth_main` does (presence of `fast_foundationstereo.engine` in the engine directory selects single-engine), and `--mode` can be set explicitly:

```bash
cpp/build/ffs_profile_speed <engine_dir> <left_image> <right_image> <intrinsic_file> \
    [--mode auto|two|single] [--warmup N] [--runs N] [--include-depth]
```

Defaults: `--mode auto`, `--warmup 10`, `--runs 30`. With `--include-depth`, the disparity-to-depth conversion is included in the timed region (otherwise only the `infer()` call is timed).

### Profile the single engine

```bash
cpp/build/ffs_profile_speed \
  output_plugin_onnx \
  demo_data/left.png demo_data/right.png demo_data/K.txt \
  --mode single --warmup 20 --runs 100
```

### Profile the two engines

```bash
cpp/build/ffs_profile_speed \
  output_two_onnx \
  demo_data/left.png demo_data/right.png demo_data/K.txt \
  --mode two --warmup 20 --runs 100
```

The output looks like:

```text
mode=single
image=960x540
model=640x480
warmup=20 runs=100
timed_region=infer
gpu  mean_ms=... p50_ms=... p90_ms=... min_ms=... max_ms=... std_ms=...
host mean_ms=... p50_ms=... p90_ms=... min_ms=... max_ms=... std_ms=...
```

- `gpu` is `cudaEventElapsedTime` between start/stop events on the inference stream (pure GPU work).
- `host` is `std::chrono::steady_clock` around the same region, including the cost of `cudaEventSynchronize`. `host >= gpu` always.

For a side-by-side comparison, run the profiler twice with different engine directories and `--mode`, then compare the `gpu mean_ms` columns.
