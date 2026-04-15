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
Run Fast FoundationStereo inference with the single ONNX model (or TRT engine)
produced by make_single_onnx.py.

Supports two backends:
  - ONNX Runtime (default if --model_file points to an .onnx, or auto-detected)
  - TensorRT     (if --model_file points to an .engine)

The model expects ImageNet-normalised inputs, so this script applies
normalisation during preprocessing.

Usage:
  # Run directly with ONNX (no trtexec step needed):
  python run_demo_single_trt.py \
      --model_dir ./output_single_onnx \
      --left_file  ../demo_data/left.png \
      --right_file ../demo_data/right.png

  # Or with an explicit model file:
  python run_demo_single_trt.py \
      --model_dir  ./output_single_onnx \
      --model_file ./output_single_onnx/fast_foundationstereo.onnx \
      --left_file  ../demo_data/left.png \
      --right_file ../demo_data/right.png
"""

import argparse
import logging
import os
import sys

import cv2
import imageio
import numpy as np
import torch
import yaml

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

from Utils import (
    set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SingleEngineTrtRunner:
    """Minimal TensorRT runner for a single engine with named I/O."""

    def __init__(self, engine_path):
        import tensorrt as trt
        self.trt = trt
        self.logger = trt.Logger(trt.Logger.WARNING)

        with open(engine_path, 'rb') as f:
            self.engine = trt.Runtime(self.logger).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError(
                f'Failed to deserialize TRT engine from {engine_path}. '
                f'This usually means the engine was built with a different '
                f'TensorRT version (yours: {trt.__version__}). '
                f'Rebuild with:  trtexec --onnx=<your .onnx> '
                f'--saveEngine={engine_path} --fp16')
        self.context = self.engine.create_execution_context()

    def _trt_to_torch_dtype(self, dt):
        trt = self.trt
        mapping = {
            trt.DataType.FLOAT:  torch.float32,
            trt.DataType.HALF:   torch.float16,
            trt.DataType.BF16:   torch.bfloat16,
            trt.DataType.INT32:  torch.int32,
            trt.DataType.INT8:   torch.int8,
            trt.DataType.BOOL:   torch.bool,
        }
        if dt not in mapping:
            raise RuntimeError(f'Unsupported TRT dtype: {dt}')
        return mapping[dt]

    def __call__(self, inputs: dict) -> dict:
        """Run inference.

        Args:
            inputs: {binding_name: torch.Tensor} for every input tensor.
        Returns:
            {binding_name: torch.Tensor} for every output tensor.
        """
        trt = self.trt

        for name, tensor in inputs.items():
            expected = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            if tensor.dtype != expected:
                inputs[name] = tensor.to(expected)
            if not inputs[name].is_contiguous():
                inputs[name] = inputs[name].contiguous()
            self.context.set_input_shape(name, tuple(inputs[name].shape))

        out_names = [
            self.engine.get_tensor_name(i)
            for i in range(self.engine.num_io_tensors)
            if self.engine.get_tensor_mode(self.engine.get_tensor_name(i))
               == trt.TensorIOMode.OUTPUT
        ]

        outputs = {}
        for name in out_names:
            shape = tuple(self.context.get_tensor_shape(name))
            dtype = self._trt_to_torch_dtype(self.engine.get_tensor_dtype(name))
            outputs[name] = torch.empty(shape, device='cuda', dtype=dtype)

        for name, tensor in inputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))
        for name, tensor in outputs.items():
            self.context.set_tensor_address(name, int(tensor.data_ptr()))

        stream = torch.cuda.current_stream().cuda_stream
        assert self.context.execute_async_v3(stream)

        return outputs


class OnnxRuntimeRunner:
    """Run inference via ONNX Runtime (GPU if available, else CPU)."""

    def __init__(self, onnx_path):
        import onnxruntime as ort
        providers = []
        if 'CUDAExecutionProvider' in ort.get_available_providers():
            providers.append('CUDAExecutionProvider')
        providers.append('CPUExecutionProvider')
        logging.info(f'ONNX Runtime providers: {providers}')
        self.session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]

    def __call__(self, inputs: dict) -> dict:
        feed = {}
        for name in self.input_names:
            tensor = inputs[name]
            if isinstance(tensor, torch.Tensor):
                tensor = tensor.cpu().float().numpy()
            feed[name] = tensor
        raw_outputs = self.session.run(self.output_names, feed)
        outputs = {}
        for name, arr in zip(self.output_names, raw_outputs):
            outputs[name] = torch.as_tensor(arr).cuda()
        return outputs


def normalize_imagenet(img_uint8: np.ndarray) -> np.ndarray:
    """Apply ImageNet normalization: (img/255 - mean) / std."""
    return ((img_uint8.astype(np.float32) / 255.0) - IMAGENET_MEAN) / IMAGENET_STD


def resolve_config(model_path: str) -> str:
    """Find the YAML config matching the model file, falling back to defaults."""
    model_dir = os.path.dirname(model_path)
    base = os.path.splitext(os.path.basename(model_path))[0]
    candidates = [
        os.path.join(model_dir, f'{base}.yaml'),
        os.path.join(model_dir, 'config.yaml'),
        os.path.join(model_dir, 'onnx.yaml'),
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        f'No .yaml config found for {model_path}. '
        'Run make_single_onnx.py first.')


def find_model(model_dir: str) -> str:
    """Find an .engine or .onnx file in the directory (prefer .engine)."""
    for ext in ('.engine', '.onnx'):
        for f in os.listdir(model_dir):
            if f.endswith(ext):
                return os.path.join(model_dir, f)
    raise FileNotFoundError(
        f'No .engine or .onnx file found in {model_dir}. '
        'Run make_single_onnx.py first.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Run Fast FoundationStereo with ONNX Runtime or TensorRT')
    parser.add_argument('--model_dir', type=str,
                        default=f'{code_dir}/output_single_onnx',
                        help='Directory containing .onnx/.engine + config.yaml')
    parser.add_argument('--model_file', type=str, default='',
                        help='Explicit path to .onnx or .engine file (overrides auto-search)')
    parser.add_argument('--left_file', type=str,
                        default=f'{code_dir}/../demo_data/left.png')
    parser.add_argument('--right_file', type=str,
                        default=f'{code_dir}/../demo_data/right.png')
    parser.add_argument('--intrinsic_file', type=str,
                        default=f'{code_dir}/../demo_data/K.txt',
                        help='Camera intrinsic matrix and baseline file')
    parser.add_argument('--out_dir', type=str,
                        default=f'{code_dir}/../output_demo')
    parser.add_argument('--remove_invisible', type=int, default=1)
    parser.add_argument('--denoise_cloud', type=int, default=1)
    parser.add_argument('--denoise_nb_points', type=int, default=30)
    parser.add_argument('--denoise_radius', type=float, default=0.03)
    parser.add_argument('--get_pc', type=int, default=1,
                        help='Generate and save point cloud')
    parser.add_argument('--zfar', type=float, default=100,
                        help='Max depth (m) to include in point cloud')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    # ── Find model and config ─────────────────────────────────────────────
    model_path = args.model_file if args.model_file else find_model(args.model_dir)
    cfg_path = resolve_config(model_path)
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    target_h, target_w = cfg['image_size']
    logging.info(f'Model target resolution: {target_h} x {target_w}')

    # ── Load model (ONNX Runtime or TensorRT) ────────────────────────────
    logging.info(f'Loading model: {model_path}')
    if model_path.endswith('.onnx'):
        runner = OnnxRuntimeRunner(model_path)
    else:
        runner = SingleEngineTrtRunner(model_path)

    # ── Read images ───────────────────────────────────────────────────────
    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)

    if img0.ndim == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]

    # ── Resize to model resolution (direct stretch) ────────────────────────
    orig_h, orig_w = img0.shape[:2]
    fx = target_w / orig_w
    fy = target_h / orig_h

    if fx != 1 or fy != 1:
        logging.info(
            f'Resizing images: {orig_h}x{orig_w} → {target_h}x{target_w} '
            f'(fx={fx:.4f}, fy={fy:.4f})')
        img0 = cv2.resize(img0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    H, W = img0.shape[:2]

    img0_ori = img0.copy()
    img1_ori = img1.copy()
    logging.info(f'Image size after resize: {img0.shape}')
    imageio.imwrite(f'{args.out_dir}/left.png', img0)
    imageio.imwrite(f'{args.out_dir}/right.png', img1)

    # ── Preprocess: ImageNet normalize → NCHW float tensor ────────────────
    img0_norm = normalize_imagenet(img0)
    img1_norm = normalize_imagenet(img1)

    t_left  = torch.as_tensor(img0_norm).cuda().float()[None].permute(0, 3, 1, 2)
    t_right = torch.as_tensor(img1_norm).cuda().float()[None].permute(0, 3, 1, 2)

    # ── Inference ─────────────────────────────────────────────────────────
    logging.info('Running inference (first run may be slow due to TRT warmup)')
    outputs = runner({'left_image': t_left, 'right_image': t_right})
    disp = outputs['disparity']
    logging.info('Inference done')

    disp = disp.float().cpu().numpy().reshape(H, W).clip(0, None) * (1.0 / fx)

    # ── Visualise disparity ──────────────────────────────────────────────
    vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
    vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
    imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)
    s = 1280 / vis.shape[1]
    resized_vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
    cv2.imshow('disp', resized_vis[:, :, ::-1])
    cv2.waitKey(0)

    # ── Remove invisible pixels ──────────────────────────────────────────
    if args.remove_invisible:
        _, xx = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        invalid = (xx - disp) < 0
        disp[invalid] = np.inf

    # ── Point cloud generation ───────────────────────────────────────────
    if args.get_pc:
        with open(args.intrinsic_file, 'r') as f:
            lines = f.readlines()
            K = (np.array(list(map(float, lines[0].rstrip().split())))
                 .astype(np.float32).reshape(3, 3))
            baseline = float(lines[1])
        K[:2] *= np.array([fx, fy], dtype=np.float32)[:, np.newaxis]
        depth = K[0, 0] * baseline / disp
        np.save(f'{args.out_dir}/depth_meter.npy', depth)

        xyz_map = depth2xyzmap(depth, K)
        pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
        pts = np.asarray(pcd.points)
        keep = (pts[:, 2] > 0) & (pts[:, 2] <= args.zfar)
        pcd = pcd.select_by_index(np.where(keep)[0])
        o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
        logging.info(f'Point cloud saved to {args.out_dir}')

        if args.denoise_cloud:
            logging.info('Denoising point cloud...')
            _, ind = pcd.remove_radius_outlier(
                nb_points=args.denoise_nb_points,
                radius=args.denoise_radius)
            pcd = pcd.select_by_index(ind)
            o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', pcd)

        logging.info('Visualizing point cloud. Press ESC to exit.')
        vis = o3d.visualization.Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
        ctr = vis.get_view_control()
        ctr.set_front([0, 0, -1])
        closest = np.asarray(pcd.points)[:, 2].argmin()
        ctr.set_lookat(np.asarray(pcd.points)[closest])
        ctr.set_up([0, -1, 0])
        vis.run()
        vis.destroy_window()
