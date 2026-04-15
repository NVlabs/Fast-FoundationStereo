import os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from omegaconf import OmegaConf
import argparse, torch, logging, yaml, time
import numpy as np
from Utils import set_logging_format, set_seed
from core.foundation_stereo import TrtRunner


def resolve_onnx_cfg_path(onnx_dir: str) -> str:
  direct = os.path.join(onnx_dir, 'onnx.yaml')
  if os.path.exists(direct):
    return direct
  parent = os.path.join(os.path.dirname(onnx_dir), 'onnx.yaml')
  if os.path.exists(parent):
    return parent
  raise FileNotFoundError(f"onnx.yaml not found in {onnx_dir} or its parent directory")


if __name__=="__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument('--model_dir', default=None, type=str)
  parser.add_argument('--hiera', default=None, type=int)
  parser.add_argument('--valid_iters', type=int, default=None, help='number of flow-field updates during forward pass (default: from onnx.yaml)')
  parser.add_argument('--max_disp', type=int, default=None, help='maximum disparity (default: from onnx.yaml)')
  parser.add_argument('--warmup', type=int, default=15, help='number of warmup iterations')
  parser.add_argument('--total', type=int, default=30, help='total number of iterations')
  parser.add_argument('--build_volume_backend', default=None, choices=['pytorch1', 'triton'], help='backend for cost-volume build (default: from onnx.yaml)')
  parser.add_argument('--onnx_dir', default=f'{code_dir}/../output', type=str, help='directory containing TensorRT engines and onnx.yaml')
  args = parser.parse_args()

  set_logging_format()
  set_seed(0)
  torch.backends.cudnn.benchmark = True
  torch.autograd.set_grad_enabled(False)

  cfg_path = resolve_onnx_cfg_path(args.onnx_dir)
  with open(cfg_path, 'r') as ff:
    cfg:dict = yaml.safe_load(ff)
  for k in args.__dict__:
    if args.__dict__[k] is not None:
      cfg[k] = args.__dict__[k]
  args = OmegaConf.create(cfg)

  model = TrtRunner(args, args.onnx_dir+'/feature_runner.engine', args.onnx_dir+'/post_runner.engine')

  H, W = int(args.image_size[0]), int(args.image_size[1])
  img0 = torch.randint(0, 256, (1, 3, H, W), dtype=torch.float32).cuda()
  img1 = torch.randint(0, 256, (1, 3, H, W), dtype=torch.float32).cuda()

  logging.info(f"TensorRT image size: {H}x{W}, warmup: {args.warmup}, total: {args.total}")

  times = []
  for i in range(args.total):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    _ = model.forward(img0, img1)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    times.append(elapsed)
    logging.info(f"Iter {i:2d}: {elapsed*1000:.1f} ms {'(warmup)' if i < args.warmup else ''}")

  measure_times = times[args.warmup:]
  avg = np.mean(measure_times) * 1000
  logging.info(f"TensorRT speed average (after warmup): {avg:.1f}[ms] over {len(measure_times)} iters")
