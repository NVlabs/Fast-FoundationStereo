import os, sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
from core.utils.utils import InputPadder
import argparse, torch, imageio, logging, yaml
import numpy as np
from Utils import (
    set_logging_format, set_seed, vis_disparity,
    depth2xyzmap, toOpen3dCloud, o3d,
)
from core.foundation_stereo import TrtRunner
import cv2
from tqdm import tqdm
import time

if __name__=="__main__":
  code_dir = os.path.dirname(os.path.realpath(__file__))
  parser = argparse.ArgumentParser()
  parser.add_argument('--onnx_dir', default=f'{code_dir}/output', type=str)
  parser.add_argument('--left_file', default=f'{code_dir}/../assets/left.png', type=str)
  parser.add_argument('--right_file', default=f'{code_dir}/../assets/right.png', type=str)
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.autograd.set_grad_enabled(False)

  with open(f'{os.path.dirname(args.onnx_dir)}/onnx.yaml', 'r') as ff:
    cfg:dict = yaml.safe_load(ff)
  for k in args.__dict__:
    if args.__dict__[k] is not None:
      cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)
  logging.info(f"args:\n{args}")
  model = TrtRunner(args, args.onnx_dir+'/feature_runner.engine', args.onnx_dir+'/post_runner.engine')

  img0 = imageio.imread(args.left_file)
  img1 = imageio.imread(args.right_file)
  if len(img0.shape)==2:
    img0 = np.tile(img0[...,None], (1,1,3))
    img1 = np.tile(img1[...,None], (1,1,3))
  img0 = img0[...,:3]
  img1 = img1[...,:3]
  H,W = img0.shape[:2]

  img0 = torch.as_tensor(img0).cuda().float()[None].permute(0,3,1,2)
  img1 = torch.as_tensor(img1).cuda().float()[None].permute(0,3,1,2)

  logging.info(f"Warming up, 1st run can be slow")
  for _ in range(5):
    disp = model.forward(img0, img1)

  start_time = time.time()
  for i in tqdm(range(1000)):
    disp = model.forward(img0, img1)

  total_time = time.time() - start_time
  average_inference_time = total_time / 1000

  logging.info(f"Total time: {total_time}")
  logging.info(f"average inference time: {average_inference_time}")

