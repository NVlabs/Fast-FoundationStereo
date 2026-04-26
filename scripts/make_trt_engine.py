"""Convert ONNX models to TensorRT engines using the Python TRT API."""
import argparse
import os
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)


def build_engine(onnx_path: str, engine_path: str, fp16: bool = True, workspace_gb: int = 4):
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_gb << 30)

    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                print(f"ONNX parse error {i}: {parser.get_error(i)}")
            raise RuntimeError(f"Failed to parse ONNX: {onnx_path}")

    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
        print("FP16 enabled")

    print(f"Building TRT engine from {onnx_path} …")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        raise RuntimeError("build_serialized_network returned None")

    with open(engine_path, "wb") as f:
        f.write(serialized)
    print(f"Engine saved -> {engine_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx_dir", type=str,
                        default="output/onnx_trt",
                        help="Directory containing feature_runner.onnx and post_runner.onnx")
    parser.add_argument("--engine_dir", type=str, default=None,
                        help="Output directory for .engine files (defaults to onnx_dir)")
    parser.add_argument("--fp16", action="store_true", default=True)
    parser.add_argument("--workspace_gb", type=int, default=4)
    args = parser.parse_args()

    engine_dir = args.engine_dir or args.onnx_dir
    os.makedirs(engine_dir, exist_ok=True)

    for name in ("feature_runner", "post_runner"):
        onnx_path = os.path.join(args.onnx_dir, f"{name}.onnx")
        engine_path = os.path.join(engine_dir, f"{name}.engine")
        build_engine(onnx_path, engine_path, fp16=args.fp16, workspace_gb=args.workspace_gb)
