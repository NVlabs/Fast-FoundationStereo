import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
import argparse, logging, yaml, imageio
import numpy as np
from Utils import (
    set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
from core.foundation_stereo import OnnxRunner
import cv2


if __name__ == '__main__':
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser(
      description=(
          'Run stereo inference using ONNX Runtime on CPU.  '
          'The ONNX models must have been exported with scripts/make_onnx.py.  '
          'NOTE: the exported models have fixed input shapes; images are '
          'resized to match the shape used at export time.'
      )
  )
  parser.add_argument('--onnx_dir', default=f'{code_dir}/output', type=str,
                      help='directory containing feature_runner.onnx and post_runner.onnx')
  parser.add_argument('--left_file',  default=f'{code_dir}/../assets/left.png',  type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  parser.add_argument('--intrinsic_file', default=f'{code_dir}/../assets/K.txt', type=str,
                      help='camera intrinsic matrix and baseline file')
  parser.add_argument('--out_dir', default='/tmp/stereo_output_onnx', type=str)
  parser.add_argument('--remove_invisible', default=1, type=int)
  parser.add_argument('--denoise_cloud',    default=1, type=int)
  parser.add_argument('--denoise_nb_points', type=int,   default=30,
                      help='number of points for radius outlier removal')
  parser.add_argument('--denoise_radius',    type=float, default=0.03,
                      help='radius for outlier removal')
  parser.add_argument('--get_pc', type=int, default=1, help='save point cloud output')
  parser.add_argument('--zfar',   type=float, default=100, help='max depth for point cloud')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)

  os.makedirs(args.out_dir, exist_ok=True)

  cfg_path = f'{os.path.dirname(args.onnx_dir)}/onnx.yaml'
  with open(cfg_path, 'r') as ff:
    cfg: dict = yaml.safe_load(ff)
  for k in args.__dict__:
    if args.__dict__[k] is not None:
      cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f'args:\n{args}')

  model = OnnxRunner(
      args,
      args.onnx_dir + '/feature_runner.onnx',
      args.onnx_dir + '/post_runner.onnx',
  )

  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  if len(img0.shape) == 2:
    img0 = np.tile(img0[..., None], (1, 1, 3))
    img1 = np.tile(img1[..., None], (1, 1, 3))
  img0 = img0[..., :3]
  img1 = img1[..., :3]
  H, W = img0.shape[:2]

  fx = args.image_size[1] / img0.shape[1]
  fy = args.image_size[0] / img0.shape[0]
  if fx != 1 or fy != 1:
    logging.warning(
        f'Resizing image to {args.image_size} (fx={fx:.4f}, fy={fy:.4f}). '
        'The ONNX models have fixed input shapes; pass images that match the '
        'size used during export to avoid resizing artefacts.'
    )
  img0 = cv2.resize(img0, fx=fx, fy=fy, dsize=None)
  img1 = cv2.resize(img1, fx=fx, fy=fy, dsize=None)
  H, W = img0.shape[:2]
  img0_ori = img0.copy()
  img1_ori = img1.copy()
  logging.info(f'img0: {img0.shape}')
  imageio.imwrite(f'{args.out_dir}/left.png',  img0)
  imageio.imwrite(f'{args.out_dir}/right.png', img1)

  # ONNX Runtime expects (B, C, H, W) float32 numpy arrays
  left_np  = img0.astype(np.float32)[None].transpose(0, 3, 1, 2)
  right_np = img1.astype(np.float32)[None].transpose(0, 3, 1, 2)

  logging.info('Start forward (first run may be slow due to torch.compile)')
  disp = model.forward(left_np, right_np)
  logging.info('Forward done')

  disp = disp.reshape(H, W).clip(0, None) * (1 / fx)

  vis = vis_disparity(disp, cmap=None)
  vis = np.concatenate([img0_ori, img1_ori, vis], axis=1)
  imageio.imwrite(f'{args.out_dir}/disp_vis.png', vis)
  s = 1280 / vis.shape[1]
  resized_vis = cv2.resize(vis, (int(vis.shape[1] * s), int(vis.shape[0] * s)))
  cv2.imshow('disp', resized_vis[:, :, ::-1])
  cv2.waitKey(0)

  if args.remove_invisible:
    yy, xx = np.meshgrid(np.arange(disp.shape[0]), np.arange(disp.shape[1]), indexing='ij')
    us_right = xx - disp
    disp[us_right < 0] = np.inf

  if args.get_pc:
    with open(args.intrinsic_file, 'r') as f:
      lines = f.readlines()
      K = np.array(list(map(float, lines[0].rstrip().split()))).astype(np.float32).reshape(3, 3)
      baseline = float(lines[1])
    K[:2] *= np.array([fx, fy])
    depth = K[0, 0] * baseline / disp
    np.save(f'{args.out_dir}/depth_meter.npy', depth)
    xyz_map = depth2xyzmap(depth, K)
    pcd = toOpen3dCloud(xyz_map.reshape(-1, 3), img0_ori.reshape(-1, 3))
    keep_mask = (np.asarray(pcd.points)[:, 2] > 0) & (np.asarray(pcd.points)[:, 2] <= args.zfar)
    keep_ids = np.arange(len(np.asarray(pcd.points)))[keep_mask]
    pcd = pcd.select_by_index(keep_ids)
    o3d.io.write_point_cloud(f'{args.out_dir}/cloud.ply', pcd)
    logging.info(f'PCL saved to {args.out_dir}')

    if args.denoise_cloud:
      logging.info('[Optional] denoising point cloud...')
      cl, ind = pcd.remove_radius_outlier(nb_points=args.denoise_nb_points, radius=args.denoise_radius)
      inlier_cloud = pcd.select_by_index(ind)
      o3d.io.write_point_cloud(f'{args.out_dir}/cloud_denoise.ply', inlier_cloud)
      pcd = inlier_cloud

    logging.info('Visualizing point cloud. Press ESC to exit.')
    vis_o3d = o3d.visualization.Visualizer()
    vis_o3d.create_window()
    vis_o3d.add_geometry(pcd)
    vis_o3d.get_render_option().point_size = 1.0
    vis_o3d.get_render_option().background_color = np.array([0.5, 0.5, 0.5])
    ctr = vis_o3d.get_view_control()
    ctr.set_front([0, 0, -1])
    id_near = np.asarray(pcd.points)[:, 2].argmin()
    ctr.set_lookat(np.asarray(pcd.points)[id_near])
    ctr.set_up([0, -1, 0])
    vis_o3d.run()
    vis_o3d.destroy_window()
