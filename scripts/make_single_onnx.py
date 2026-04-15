# SPDX-FileCopyrightText: NVIDIA CORPORATION & AFFILIATES
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""
Export Fast FoundationStereo as a **single** ONNX model.

Unlike make_onnx.py (which splits into feature_runner + post_runner with a
Triton GWC kernel in between), this script produces one self-contained ONNX
that can be converted to a single TensorRT engine via trtexec.

Key design choices:
  - The GWC and concat cost volumes are built with ONNX-compatible ops
    (pad + slice + stack).  The upstream pytorch1 variants use
    Tensor.unfold / torch.flip which the ONNX exporter cannot handle.
  - ImageNet normalization is STRIPPED from the model so that it can be
    applied externally (e.g. via Isaac ROS ImageNormalizeNode).  The ONNX
    model expects **pre-normalized** float inputs:
        pixel = (pixel_0_255 - mean) / std
        mean  = [123.675, 116.28, 103.53]   (ImageNet, in 0-255 scale)
        std   = [ 58.395, 57.12, 57.375]
  - Inputs:  left_image  (1, 3, H, W)  float32, ImageNet-normalised
             right_image (1, 3, H, W)  float32, ImageNet-normalised
  - Output:  disparity   (1, 1, H, W)  float32

Usage:
  python make_single_onnx.py \\
      --model_dir ../weights/model_best_bp2_serialize.pth \\
      --save_path ./output_single_onnx --height 480 --width 640

  # Then build a TensorRT engine:
  trtexec --onnx=./output_single_onnx/fast_foundationstereo.onnx \\
      --saveEngine=fast_foundationstereo.engine --fp16
"""

import argparse
import logging
import os
import sys

os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'

code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')

import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import core.foundation_stereo as _fs_module
from core.foundation_stereo import FastFoundationStereo


# ---------------------------------------------------------------------------
# ONNX-compatible cost-volume builders
#
# The upstream *_optimized_pytorch1 variants use Tensor.unfold + torch.flip
# which the ONNX tracer cannot export.  These replacements build the
# disparity-shifted target volume with an explicit loop over disparities
# using only F.pad, slicing, and torch.stack — all fully ONNX-exportable.
# ---------------------------------------------------------------------------

def _build_gwc_volume_onnx(refimg_fea, targetimg_fea, maxdisp,
                           num_groups, normalize=True):
    dtype = refimg_fea.dtype
    B, C, H, W = refimg_fea.shape
    channels_per_group = C // num_groups

    ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)

    shifted = [
        F.pad(targetimg_fea, (d, 0, 0, 0), 'constant', 0.0)[:, :, :, :W]
        for d in range(maxdisp)
    ]
    target_volume = torch.stack(shifted, dim=2)

    ref_volume = ref_volume.view(B, num_groups, channels_per_group,
                                 maxdisp, H, W)
    target_volume = target_volume.view(B, num_groups, channels_per_group,
                                       maxdisp, H, W)

    if normalize:
        ref_volume = F.normalize(ref_volume.float(), dim=2).to(dtype)
        target_volume = F.normalize(target_volume.float(), dim=2).to(dtype)

    return (ref_volume * target_volume).sum(dim=2).contiguous()


def _build_concat_volume_onnx(refimg_fea, targetimg_fea, maxdisp):
    B, C, H, W = refimg_fea.shape

    ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)

    shifted = [
        F.pad(targetimg_fea, (d, 0, 0, 0), 'constant', 0.0)[:, :, :, :W]
        for d in range(maxdisp)
    ]
    target_volume = torch.stack(shifted, dim=2)

    return torch.cat((ref_volume, target_volume), dim=1).contiguous()


# ---------------------------------------------------------------------------

class FastFoundationStereoSingleOnnx(nn.Module):
    """Thin wrapper that calls the full model with ONNX-compatible settings.

    Before ONNX tracing the caller must monkey-patch:
      - normalize_image        → identity  (normalization done externally)
      - build_gwc_volume_*     → _build_gwc_volume_onnx
      - build_concat_volume_*  → _build_concat_volume_onnx
    """

    def __init__(self, model: FastFoundationStereo):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, left_image, right_image):
        return self.model.forward(
            left_image, right_image,
            iters=self.model.args.valid_iters,
            test_mode=True,
            optimize_build_volume='pytorch1',
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Export Fast FoundationStereo as a single ONNX model')
    parser.add_argument(
        '--model_dir', type=str,
        default=f'{code_dir}/../weights/model_best_bp2_serialize.pth',
        help='Path to the serialized .pth model')
    parser.add_argument(
        '--save_path', type=str,
        default=f'{code_dir}/output_single_onnx',
        help='Directory to save the ONNX model and config')
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--valid_iters', type=int, default=8,
                        help='GRU refinement iterations')
    parser.add_argument('--max_disp', type=int, default=192,
                        help='Maximum disparity (in pixels at full resolution)')
    parser.add_argument('--onnx_name', type=str, default='fast_foundationstereo',
                        help='Base name for the saved ONNX file (without .onnx extension)')
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s')

    assert args.height % 32 == 0 and args.width % 32 == 0, \
        'height and width must be divisible by 32'

    os.makedirs(args.save_path, exist_ok=True)
    torch.autograd.set_grad_enabled(False)

    if not os.path.isfile(args.model_dir):
        raise FileNotFoundError(f'Model file not found: {args.model_dir}')

    logging.info(f'Loading model from {args.model_dir}')
    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.args.mixed_precision = False
    model.cuda().eval()

    wrapper = FastFoundationStereoSingleOnnx(model)
    wrapper.cuda().eval()

    left_img = torch.randn(1, 3, args.height, args.width, device='cuda')
    right_img = torch.randn(1, 3, args.height, args.width, device='cuda')

    onnx_name = args.onnx_name if args.onnx_name.endswith('.onnx') else f'{args.onnx_name}.onnx'
    onnx_path = os.path.join(args.save_path, onnx_name)
    logging.info(f'Exporting ONNX ({args.height}x{args.width}) → {onnx_path}')

    # Monkey-patch non-ONNX-exportable functions before tracing
    _fs_module.normalize_image = lambda img: img
    _fs_module.build_gwc_volume_optimized_pytorch1 = _build_gwc_volume_onnx
    _fs_module.build_concat_volume_optimized_pytorch1 = _build_concat_volume_onnx

    torch.onnx.export(
        wrapper,
        (left_img, right_img),
        onnx_path,
        opset_version=17,
        input_names=['left_image', 'right_image'],
        output_names=['disparity'],
        do_constant_folding=True,
    )

    cfg = OmegaConf.to_container(model.args)
    cfg['image_size'] = [args.height, args.width]
    config_name = os.path.splitext(onnx_name)[0] + '.yaml'
    config_path = os.path.join(args.save_path, config_name)
    with open(config_path, 'w') as f:
        yaml.safe_dump(cfg, f)

    logging.info(f'ONNX model  : {onnx_path}')
    logging.info(f'Config      : {config_path}')
    logging.info(f'Resolution  : {args.height} x {args.width}')
    logging.info(
        f'Build TRT engine:\n'
        f'  trtexec --onnx={onnx_path} '
        f'--saveEngine={args.save_path}/{os.path.splitext(onnx_name)[0]}.engine --fp16')
