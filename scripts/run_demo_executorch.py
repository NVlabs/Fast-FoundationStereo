"""
Run the ExecuTorch program made by scripts/make_executorch.py on a stereo pair.

With --compare_model_dir, also runs the PyTorch model on the same input and prints the
largest disparity difference and the end-point error (mean absolute difference, in pixels).
"""

import argparse
import logging
import os
import sys
from pathlib import Path

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')

import cv2
import imageio
import numpy as np
import torch
import yaml
from Utils import set_logging_format, set_seed, vis_disparity


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default=f'{code_dir}/../output_executorch', help='Directory with the .pte and .yaml from make_executorch.py')
    parser.add_argument('--left_file', type=str, default=f'{code_dir}/../demo_data/left.png')
    parser.add_argument('--right_file', type=str, default=f'{code_dir}/../demo_data/right.png')
    parser.add_argument('--out_dir', type=str, default=f'{code_dir}/../output_demo_executorch')
    parser.add_argument('--compare_model_dir', type=str, default='', help='Optional .pth checkpoint to compare against')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)
    torch.autograd.set_grad_enabled(False)
    os.makedirs(args.out_dir, exist_ok=True)

    import torch_tensorrt_executorch_runtime  # noqa: F401  (registers the TensorRT delegate)
    from executorch.runtime import Runtime

    with open(f'{args.model_dir}/fast_foundationstereo.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    target_h, target_w = cfg['image_size']

    img0 = imageio.imread(args.left_file)
    img1 = imageio.imread(args.right_file)
    if img0.ndim == 2:
        img0 = np.tile(img0[..., None], (1, 1, 3))
        img1 = np.tile(img1[..., None], (1, 1, 3))
    img0 = img0[..., :3]
    img1 = img1[..., :3]
    if img0.shape[:2] != (target_h, target_w):
        logging.info(f'Resizing images from {img0.shape[:2]} to {(target_h, target_w)}, the size the program was exported for')
        img0 = cv2.resize(img0, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        img1 = cv2.resize(img1, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    left = torch.as_tensor(img0).float()[None].permute(0, 3, 1, 2).contiguous()
    right = torch.as_tensor(img1).float()[None].permute(0, 3, 1, 2).contiguous()

    # The CUDA backend keeps its data in aoti_cuda_blob.ptd next to the .pte.
    ptd = Path(args.model_dir) / 'aoti_cuda_blob.ptd'
    program = Runtime.get().load_program(Path(args.model_dir) / 'fast_foundationstereo.pte',
                                         data_path=ptd if ptd.exists() else None)
    forward = program.load_method('forward')
    disp = forward.execute((left, right))[0].float().reshape(target_h, target_w)
    logging.info('ExecuTorch forward done')

    if args.compare_model_dir:
        model = torch.load(args.compare_model_dir, map_location='cpu', weights_only=False)
        for k in ('max_disp', 'valid_iters', 'mixed_precision'):
            model.args[k] = cfg[k]
        model.cuda().eval()
        ref = model.forward(left.cuda(), right.cuda(), iters=cfg['valid_iters'], test_mode=True,
                            optimize_build_volume=cfg['build_volume'])
        diff = (disp - ref.float().cpu().reshape(target_h, target_w)).abs()
        logging.info(f'ExecuTorch vs PyTorch: max abs diff {diff.max().item():.4f} px, EPE {diff.mean().item():.4f} px, '
                     f'pixels off by more than 1 px: {(diff > 1).float().mean().item() * 100:.3f}%')

    # Disparity in pixels of the resized images, the size the program was exported for.
    disp = disp.numpy().clip(0, None)
    np.save(f'{args.out_dir}/disp.npy', disp)
    vis = vis_disparity(disp, color_map=cv2.COLORMAP_TURBO)
    imageio.imwrite(f'{args.out_dir}/disp_vis.png', np.concatenate([img0, img1, vis], axis=1))
    logging.info(f'Saved results to {args.out_dir}')
