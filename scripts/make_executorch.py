"""
Export Fast FoundationStereo as a single ExecuTorch program (.pte).

The whole model goes through torch.export, with no ONNX step and no TensorRT plugin.
TensorRT builds engines for the parts it supports. With --build_volume triton (default),
the Triton GWC kernel runs through ExecuTorch's CUDA backend, in the same .pte file.
With --build_volume pytorch1, the cost volume is plain PyTorch and TensorRT takes all of it.

The program takes raw RGB images in 0-255 (float32, shape 1x3xHxW), like scripts/run_demo.py,
and returns the disparity (1x1xHxW). Run it with scripts/run_demo_executorch.py.

Requires: pip install "torch_tensorrt[executorch]", and a CUDA toolkit (nvcc) for the CUDA backend.
"""

import argparse
import logging
import os
import sys

# As in the ONNX export scripts: run the @torch.compile helpers as plain PyTorch.
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')

import torch
import torch.nn as nn
import yaml
from omegaconf import OmegaConf


class FastFoundationStereoExport(nn.Module):
    def __init__(self, model, build_volume):
        super().__init__()
        self.model = model
        self.build_volume = build_volume

    def forward(self, left, right):
        return self.model.forward(left, right, iters=self.model.args.valid_iters, test_mode=True,
                                  optimize_build_volume=self.build_volume)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Fast FoundationStereo as an ExecuTorch .pte')
    parser.add_argument('--model_dir', type=str, default=f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth')
    parser.add_argument('--save_path', type=str, default=f'{code_dir}/../output_executorch', help='Directory to save the .pte and its config')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--valid_iters', type=int, default=8)
    parser.add_argument('--max_disp', type=int, default=192)
    parser.add_argument('--build_volume', type=str, default='triton', choices=['triton', 'pytorch1'])
    parser.add_argument('--mixed_precision', type=int, default=0, help='0: FP32, 1: FP16 where the PyTorch model uses autocast (faster, less exact)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    assert args.height % 32 == 0 and args.width % 32 == 0, 'height and width must be divisible by 32'
    os.makedirs(args.save_path, exist_ok=True)
    torch.autograd.set_grad_enabled(False)

    import torch_tensorrt
    from executorch.backends.cuda.cuda_backend import CudaBackend
    from executorch.backends.cuda.cuda_partitioner import CudaPartitioner

    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.args.mixed_precision = bool(args.mixed_precision)
    model.cuda().eval()
    wrapper = FastFoundationStereoExport(model, args.build_volume).eval()

    left = torch.randint(0, 256, (1, 3, args.height, args.width), device='cuda').float()
    right = torch.randint(0, 256, (1, 3, args.height, args.width), device='cuda').float()

    logging.info('torch.export')
    exported = torch.export.export(wrapper, (left, right))
    logging.info('Building TensorRT engines, this can take 10 minutes or more')
    trt_gm = torch_tensorrt.dynamo.compile(exported, arg_inputs=[left, right], min_block_size=1)

    pte_path = os.path.join(args.save_path, 'fast_foundationstereo.pte')
    # TensorRT takes what it can, and the CUDA backend compiles the rest (the Triton kernel).
    torch_tensorrt.save(trt_gm, pte_path, output_format='executorch', retrace=False,
                        partitioners=[CudaPartitioner([CudaBackend.generate_method_name_compile_spec('forward')])])

    cfg = OmegaConf.to_container(model.args)
    cfg['image_size'] = [args.height, args.width]
    cfg['build_volume'] = args.build_volume
    with open(os.path.join(args.save_path, 'fast_foundationstereo.yaml'), 'w') as f:
        yaml.safe_dump(cfg, f)
    logging.info(f'Saved {pte_path}')
