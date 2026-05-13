# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, imageio, logging, yaml
import numpy as np
from Utils import (
    AMP_DTYPE, set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
import cv2


if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()

  parser.add_argument('--model_dir', default=f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth', type=str)
  # parser.add_argument('--left_file', default=f'{code_dir}/../data/mono/img0074.jpg', type=str)
  # parser.add_argument('--right_file', default=f'{code_dir}/../data/mono/img0076.jpg', type=str)  
  parser.add_argument('--left_file', default=f'{code_dir}/../demo_data/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../demo_data/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../demo_data/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--out_dir', default=f'{code_dir}/../demo_data_out', type=str)
  parser.add_argument('--remove_invisible', default=1, type=int)
  parser.add_argument('--denoise_cloud', default=0, type=int)
  parser.add_argument('--denoise_nb_points', type=int, default=30, help='number of points to consider for radius outlier removal')
  parser.add_argument('--denoise_radius', type=float, default=0.03, help='radius to use for outlier removal')
  parser.add_argument('--scale', default=1, type=float)
  parser.add_argument('--hiera', default=0, type=int)
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
  parser.add_argument('--max_disp', type=int, default=192, help='maximum disparity')
  parser.add_argument('--zfar', type=float, default=100, help="max depth to include in point cloud")
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  #os.system(f'rm -rf {args.out_dir} && mkdir -p {args.out_dir}')

  with open(f'{os.path.dirname(args.model_dir)}/cfg.yaml', 'r') as ff:
    cfg:dict = yaml.safe_load(ff)
  for k in args.__dict__:
    if args.__dict__[k] is not None:
      cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
  model.args.valid_iters = args.valid_iters
  model.args.max_disp = args.max_disp

  model.cuda().eval()

  scale = args.scale

  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  if len(img0.shape)==2:
    img0 = np.tile(img0[...,None], (1,1,3))
    img1 = np.tile(img1[...,None], (1,1,3))
  img0 = img0[...,:3]
  img1 = img1[...,:3]
  H,W = img0.shape[:2]

  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
  H,W = img0.shape[:2]
  img0_ori = img0.copy()
  img1_ori = img1.copy()
  logging.info(f"img0: {img0.shape}")
  # imageio.imwrite(f'{args.out_dir}/left.png', img0)
  # imageio.imwrite(f'{args.out_dir}/right.png', img1)

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)
  padder = InputPadder(img0.shape, divis_by=32, force_square=False)
  img0, img1 = padder.pad(img0, img1)

  logging.info(f"Start forward, 1st time run can be slow due to compilation")
  with torch.amp.autocast('cuda', enabled=True, dtype=AMP_DTYPE):
    if not args.hiera:
      disp = model.forward(img0, img1, iters=args.valid_iters, test_mode=True, optimize_build_volume='pytorch1')
    else:
      disp = model.run_hierachical(img0, img1, iters=args.valid_iters, test_mode=True, small_ratio=0.5)
  logging.info("forward done")
  disp = padder.unpad(disp.float())
  disp = disp.data.cpu().numpy().reshape(H,W).clip(0, None)

  cmap = None
  min_val = None
  max_val = None
  vis = vis_disparity(disp, min_val=min_val, max_val=max_val, cmap=cmap, color_map=cv2.COLORMAP_TURBO)
  vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/disp_0074.png', vis)
  s = 1280/vis.shape[1]
  resized_vis = cv2.resize(vis, (int(vis.shape[1]*s), int(vis.shape[0]*s)))
  cv2.imshow('disp', resized_vis[:,:,::-1])
  cv2.waitKey(0)

  if args.remove_invisible:
    yy,xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx-disp
    invalid = us_right<0
    disp[invalid] = np.inf

  if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3,3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0,0]*baseline/disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1,3), img0_ori.reshape(-1,3))
    keep_mask = (np.asarray(pcd.points)[:,2]>0) & (np.asarray(pcd.points)[:,2]<=args.zfar)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
    logging.info(f"PCL saved to {args.out_dir}")

    if args.denoise_cloud:
      logging.info("[Optional step] denoise point cloud...")
      pcd = pcd.voxel_down_sample(voxel_size=0.001)
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
      pcd = inlier_cloud

    logging.info("Visualizing point cloud. Press ESC to exit.")
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().point_size = 1.0
    vis.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    ctr = vis.get_view_control()
    ctr.set_front([0, 0, -1])
    id = np.asarray(pcd.points)[:,2].argmin()
    ctr.set_lookat(np.asarray(pcd.points)[id])
    ctr.set_up([0, -1, 0])
    vis.run()
    vis.destroy_window()
