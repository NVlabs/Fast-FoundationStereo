import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
import argparse, imageio, logging, yaml
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from core.utils.utils import InputPadder
from Utils import (
    set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
import cv2


def build_gwc_volume(refimg_fea, targetimg_fea, maxdisp, num_groups, normalize=True):
  """Pure numpy/torch GWC volume construction matching build_gwc_volume_optimized_pytorch1."""
  dtype = refimg_fea.dtype
  B, C, H, W = refimg_fea.shape
  channels_per_group = C // num_groups

  ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
  padded_target = F.pad(targetimg_fea, (maxdisp - 1, 0, 0, 0))
  unfolded_target = padded_target.unfold(3, W, 1)
  target_volume = torch.flip(unfolded_target, [3]).permute(0, 1, 3, 2, 4)
  ref_volume = ref_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
  target_volume = target_volume.view(B, num_groups, channels_per_group, maxdisp, H, W)
  if normalize:
    ref_volume = F.normalize(ref_volume.float(), dim=2).to(dtype)
    target_volume = F.normalize(target_volume.float(), dim=2).to(dtype)

  cost_volume = (ref_volume * target_volume).sum(dim=2)
  return cost_volume.contiguous()


if __name__ == "__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--onnx_dir', default=f'{code_dir}/../output', type=str)
  parser.add_argument('--left_file', default=f'{code_dir}/../demo_data/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../demo_data/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../demo_data/K.txt', type=str, help='camera intrinsic matrix and baseline file')
  parser.add_argument('--out_dir', default='output_onnx', type=str)
  parser.add_argument('--remove_invisible', default=1, type=int)
  parser.add_argument('--denoise_cloud', default=0, type=int)
  parser.add_argument('--denoise_nb_points', type=int, default=30)
  parser.add_argument('--denoise_radius', type=float, default=0.03)
  parser.add_argument('--scale', default=1, type=float)
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--zfar', type=float, default=100)
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  os.makedirs(args.out_dir, exist_ok=True)

  # Load config from onnx.yaml
  with open(f'{args.onnx_dir}/onnx.yaml', 'r') as ff:
    cfg = yaml.safe_load(ff)
  for k in args.__dict__:
    if args.__dict__[k] is not None:
      cfg[k] = args.__dict__[k]
  cfg = OmegaConf.create(cfg)
  logging.info(f"args:\n{cfg}")

  max_disp = cfg.max_disp
  cv_group = cfg.get('cv_group', 8)
  normalize = cfg.get('normalize', True)

  # Load ONNX sessions
  providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
  logging.info("Loading ONNX models...")
  feature_session = ort.InferenceSession(f'{args.onnx_dir}/feature_runner.onnx', providers=providers)
  post_session = ort.InferenceSession(f'{args.onnx_dir}/post_runner.onnx', providers=providers)
  logging.info(f"Using provider: {feature_session.get_providers()}")

  # Load and preprocess images
  scale = args.scale
  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  if len(img0.shape) == 2:
    img0 = np.tile(img0[..., None], (1, 1, 3))
    img1 = np.tile(img1[..., None], (1, 1, 3))
  img0 = img0[..., :3]
  img1 = img1[..., :3]

  img0 = cv2.resize(img0, fx=scale, fy=scale, dsize=None)
  img1 = cv2.resize(img1, dsize=(img0.shape[1], img0.shape[0]))
  H, W = img0.shape[:2]
  img0_ori = img0.copy()
  img1_ori = img1.copy()
  logging.info(f"img0: {img0.shape}")
  imageio.imwrite(f'{args.out_dir}/left.png', img0)
  imageio.imwrite(f'{args.out_dir}/right.png', img1)

  # Convert to NCHW float32 and pad to be divisible by 32
  img0_t = torch.as_tensor(img0).float()[None].permute(0, 3, 1, 2)  # (1,3,H,W)
  img1_t = torch.as_tensor(img1).float()[None].permute(0, 3, 1, 2)
  padder = InputPadder(img0_t.shape, divis_by=32, force_square=False)
  img0_t, img1_t = padder.pad(img0_t, img1_t)
  logging.info(f"Padded shape: {img0_t.shape}")
  img0_np = img0_t.numpy()
  img1_np = img1_t.numpy()

  # Run feature extraction
  logging.info("Running feature extraction...")
  feat_outputs = feature_session.run(None, {'left': img0_np, 'right': img1_np})
  feat_names = [o.name for o in feature_session.get_outputs()]
  feat_dict = dict(zip(feat_names, feat_outputs))

  features_left_04 = torch.from_numpy(feat_dict['features_left_04']).cuda()
  features_right_04 = torch.from_numpy(feat_dict['features_right_04']).cuda()

  # Build GWC volume (between the two ONNX models)
  logging.info("Building GWC volume...")
  gwc_volume = build_gwc_volume(features_left_04.half(), features_right_04.half(), max_disp // 4, cv_group, normalize=normalize)

  # Prepare post_runner inputs, matching expected dtypes
  onnx_to_np_dtype = {'tensor(float)': np.float32, 'tensor(float16)': np.float16}
  post_inputs = {}
  for inp in post_session.get_inputs():
    np_dtype = onnx_to_np_dtype.get(inp.type, np.float32)
    if inp.name == 'gwc_volume':
      post_inputs[inp.name] = gwc_volume.cpu().numpy().astype(np_dtype)
    elif inp.name in feat_dict:
      post_inputs[inp.name] = feat_dict[inp.name].astype(np_dtype)
    else:
      raise RuntimeError(f"Unknown post_runner input: {inp.name}")

  # Run post processing
  logging.info("Running post processing...")
  post_outputs = post_session.run(None, post_inputs)
  disp = torch.from_numpy(post_outputs[0])  # (1,1,Hp,Wp)
  disp = padder.unpad(disp.float())
  disp = disp.numpy().reshape(H, W).clip(0, None)
  logging.info("Forward done")

  vis = vis_disparity(disp)
  vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)
  s = 1280 / vis.shape[1]
  resized_vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
  cv2.imshow('disp', resized_vis[:, :, ::-1])
  cv2.waitKey(0)

  if args.remove_invisible:
    yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx - disp
    invalid = us_right < 0
    disp[invalid] = np.inf

  if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
      baseline = float(lines[1])
    K[:2] *= scale
    depth = K[0, 0] * baseline / disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
    keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.zfar)
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
    id = np.asarray(pcd.points)[:, 2].argmin()
    ctr.set_lookat(np.asarray(pcd.points)[id])
    ctr.set_up([0, -1, 0])
    vis.run()
    vis.destroy_window()
