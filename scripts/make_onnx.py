import warnings, argparse, logging, os, sys,zipfile
import torch.nn as nn
os.environ['TORCH_COMPILE_DISABLE'] = '1'
os.environ['TORCHDYNAMO_DISABLE'] = '1'
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(f'{code_dir}/../')
import omegaconf, yaml, torch,pdb
from omegaconf import OmegaConf
from core.foundation_stereo import FastFoundationStereo, TrtFeatureRunner, TrtPostRunner, build_gwc_volume_triton
import Utils as U


class FoundationStereoOnnx(FastFoundationStereo):
    def __init__(self, args):
        super().__init__(args)

    @torch.no_grad()
    def forward(self, left, right):
        """ Removes extra outputs and hyper-parameters """
        with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
            disp = FastFoundationStereo.forward(self, left, right, iters=self.args.valid_iters, test_mode=True, optimize_build_volume=False)
        return disp


class SingleOnnxRunner(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @torch.no_grad()
    def forward(self, left, right):
        with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
            return self.model(
                left,
                right,
                iters=self.model.args.valid_iters,
                test_mode=True,
                optimize_build_volume='pytorch1',
            )



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument('--model_dir', type=str, default=f'{code_dir}/../weights/model_best_bp2_serialize.pth')
    parser.add_argument('--save_path', type=str, default=f'/home/bowen/debug/', help='Path to save results.')
    parser.add_argument('--height', type=int, default=448)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--mixed_precision', default=True, action='store_true', help='use mixed precision')
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--n_gru_layers', type=int, default=1, help="number of hidden GRU levels")
    parser.add_argument('--max_disp', type=int, default=192, help="max disp of geometry encoding volume")
    parser.add_argument('--low_memory', type=int, default=1, help='reduce memory usage')
    parser.add_argument('--single_onnx', action='store_true', help='Export the full model to a single ONNX file using the pure PyTorch volume builder')
    parser.add_argument('--single_onnx_name', type=str, default='foundation_stereo.onnx', help='Filename for the single-model ONNX export')
    args = parser.parse_args()
    os.makedirs(args.save_path, exist_ok=True)

    torch.autograd.set_grad_enabled(False)

    model = torch.load(args.model_dir, map_location='cpu', weights_only=False)
    model.args.max_disp = args.max_disp
    model.args.valid_iters = args.valid_iters
    model.args.image_size = [args.height, args.width]
    model.cuda().eval()

    feature_runner = TrtFeatureRunner(model)
    post_runner = TrtPostRunner(model)

    feature_runner.cuda().eval()
    post_runner.cuda().eval()
    assert args.height % 32 == 0 and args.width % 32 == 0, "height and width must be divisible by 32"
    left_img = torch.randn(1, 3, args.height, args.width).cuda().float()*255
    right_img = torch.randn(1, 3, args.height, args.width).cuda().float()*255

    if args.single_onnx:
      single_runner = SingleOnnxRunner(model).cuda().eval()
      torch.onnx.export(
          single_runner,
          (left_img, right_img),
          os.path.join(args.save_path, args.single_onnx_name),
          opset_version=17,
          input_names=['left', 'right'],
          output_names=['disp'],
          do_constant_folding=True,
      )
    else:
      torch.onnx.export(
          feature_runner,
          (left_img, right_img),
          args.save_path+'/feature_runner.onnx',
          opset_version=17,
          input_names = ['left', 'right'],
          output_names = ['features_left_04', 'features_left_08', 'features_left_16', 'features_left_32', 'features_right_04', 'stem_2x'],
          do_constant_folding=True
      )

      features_left_04, features_left_08, features_left_16, features_left_32, features_right_04, stem_2x = feature_runner(left_img, right_img)
      gwc_volume = build_gwc_volume_triton(features_left_04.half(), features_right_04.half(), args.max_disp//4, model.cv_group)
      disp = post_runner(features_left_04.float(), features_left_08.float(), features_left_16.float(), features_left_32.float(), features_right_04.float(), stem_2x.float(), gwc_volume.float())

      torch.onnx.export(
          post_runner,
          (features_left_04, features_left_08, features_left_16, features_left_32, features_right_04, stem_2x, gwc_volume),
          args.save_path+'/post_runner.onnx',
          opset_version=17,
          input_names = ['features_left_04', 'features_left_08', 'features_left_16', 'features_left_32', 'features_right_04', 'stem_2x', 'gwc_volume'],
          output_names = ['disp'],
          do_constant_folding=True
      )

    with open(f'{args.save_path}/onnx.yaml', 'w') as f:
      yaml.safe_dump(OmegaConf.to_container(model.args), f)
