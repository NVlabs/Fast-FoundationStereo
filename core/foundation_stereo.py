import torch,pdb,logging,timm
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys,os
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.update import BasicSelectiveMultiUpdateBlock
from core.extractor import ContextNetSharedBackbone, Feature
from core.geometry import Combined_Geo_Encoding_Volume
from core.submodule import (
    BasicConv, Conv3dNormActReduced, ResnetBasicBlock3D, BasicConv_IN, Conv2x,
    FeatureAtt, CostVolumeDisparityAttention, SpatialAttentionExtractor,
    ChannelAttentionEnhancement, disparity_regression, context_upsample,
    build_gwc_volume_optimized_pytorch1, build_gwc_volume_triton,
    build_concat_volume_optimized_pytorch1, build_concat_volume_optimized_pytorch,
)
from core.utils.utils import InputPadder
import Utils as U
import time

sys.modules['foundation_stereo_ori'] = sys.modules['core']
sys.modules['foundation_stereo_ori.submodule'] = sys.modules['core.submodule']
sys.modules['foundation_stereo_ori.extractor'] = sys.modules['core.extractor']
sys.modules['foundation_stereo_ori.update'] = sys.modules['core.update']
sys.modules['foundation_stereo_ori.foundation_stereo'] = sys.modules['core.foundation_stereo']
class FoundationStereo(nn.Module):
  pass



def normalize_image(img):
    '''
    @img: (B,C,H,W) in range 0-255, RGB order
    '''
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    return (img/255.0 - mean) / std


class hourglass(nn.Module):
    def __init__(self, cfg, in_channels, feat_dims=None):
        super().__init__()
        self.cfg = cfg
        self.conv1 = nn.Sequential(BasicConv(in_channels, in_channels*2, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.conv2 = nn.Sequential(BasicConv(in_channels*2, in_channels*4, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17))

        self.conv3 = nn.Sequential(BasicConv(in_channels*4, in_channels*6, is_3d=True, bn=True, relu=True, kernel_size=3,
                                             padding=1, stride=2, dilation=1),
                                   Conv3dNormActReduced(in_channels*6, in_channels*6, kernel_size=3, kernel_disp=17))


        self.conv3_up = BasicConv(in_channels*6, in_channels*4, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv2_up = BasicConv(in_channels*4, in_channels*2, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))

        self.conv1_up = BasicConv(in_channels*2, in_channels, deconv=True, is_3d=True, bn=True,
                                  relu=True, kernel_size=(4, 4, 4), padding=(1, 1, 1), stride=(2, 2, 2))
        self.conv_out = nn.Sequential(
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
          Conv3dNormActReduced(in_channels, in_channels, kernel_size=3, kernel_disp=17),
        )

        self.agg_0 = nn.Sequential(BasicConv(in_channels*8, in_channels*4, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*4, in_channels*4, kernel_size=3, kernel_disp=17),)

        self.agg_1 = nn.Sequential(BasicConv(in_channels*4, in_channels*2, is_3d=True, kernel_size=1, padding=0, stride=1),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17),
                                   Conv3dNormActReduced(in_channels*2, in_channels*2, kernel_size=3, kernel_disp=17))

        self.atts = nn.ModuleDict({
          "4": CostVolumeDisparityAttention(d_model=in_channels, nhead=4, dim_feedforward=in_channels, norm_first=False, num_transformer=4, max_len=self.cfg['max_disp']//16),
        })
        self.conv_patch = nn.Sequential(
          nn.Conv3d(in_channels, in_channels, kernel_size=4, stride=4, padding=0, groups=in_channels),
          nn.BatchNorm3d(in_channels),
        )
        self.feature_att_8 = FeatureAtt(in_channels*2, feat_dims[1])
        self.feature_att_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_32 = FeatureAtt(in_channels*6, feat_dims[3])
        self.feature_att_up_16 = FeatureAtt(in_channels*4, feat_dims[2])
        self.feature_att_up_8 = FeatureAtt(in_channels*2, feat_dims[1])

        self.post32_to_16 = None
        self.post16_to_8 = None
        self.post8_to_4 = None

    def forward(self, x, features):
        conv1 = self.conv1(x)
        conv1 = self.feature_att_8(conv1, features[1])

        conv2 = self.conv2(conv1)
        conv2 = self.feature_att_16(conv2, features[2])

        conv3 = self.conv3(conv2)
        conv3 = self.feature_att_32(conv3, features[3])
        if not hasattr(self, 'post32_to_16') or self.post32_to_16 is None:
          conv3_up = self.conv3_up(conv3)
          conv2 = torch.cat((conv3_up, conv2), dim=1)
          conv2 = self.agg_0(conv2)
          conv2 = self.feature_att_up_16(conv2, features[2])
        else:
          conv2 = self.post32_to_16(conv2, conv3, features[2])

        if not hasattr(self, 'post16_to_8') or self.post16_to_8 is None:
          conv2_up = self.conv2_up(conv2)
          conv1 = torch.cat((conv2_up, conv1), dim=1)
          conv1 = self.agg_1(conv1)
          conv1 = self.feature_att_up_8(conv1, features[1])
        else:
          conv1 = self.post16_to_8(conv1, conv2, features[1])

        conv = self.conv1_up(conv1)
        if not hasattr(self, 'post8_to_4') or self.post8_to_4 is None:
          x = self.conv_patch(x)
          x = self.atts["4"](x)
          x = F.interpolate(x, scale_factor=4, mode='trilinear', align_corners=False)
          conv = conv + x
          conv = self.conv_out(conv)
        else:
          conv = self.post8_to_4(x, conv)

        return conv


class FastFoundationStereo(nn.Module):
  def __init__(self, args):
    super().__init__()
    self.args = args
    self.dtype = torch.float32

    context_dims = args.hidden_dims
    self.cv_group = args.get('cv_group', 8)
    self.concat_channel = 24
    volume_dim = args.get('volume_dim', 28)
    self.volume_dim = volume_dim
    self.update_block = BasicSelectiveMultiUpdateBlock(self.args, self.args.hidden_dims[0], volume_dim=volume_dim)
    self.sam = SpatialAttentionExtractor()
    self.cam = ChannelAttentionEnhancement(self.args.hidden_dims[0])
    self.context_zqr_convs = nn.ModuleList([nn.Conv2d(context_dims[i], args.hidden_dims[i]*3, kernel_size=3, padding=3//2) for i in range(self.args.n_gru_layers)])
    self.feature = Feature(args)
    self.proj_cmb = nn.Conv2d(self.feature.d_out[0], self.concat_channel//2, kernel_size=1, padding=0)
    self.cnet = ContextNetSharedBackbone(args, c04=self.feature.d_out[0], c08=self.feature.d_out[1], c16=self.feature.d_out[2], output_dim=[args.hidden_dims, context_dims])

    self.stem_2 = nn.Sequential(
      BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
      nn.Conv2d(32, 32, 3, 1, 1, bias=False),
      nn.InstanceNorm2d(32), nn.ReLU()
    )
    self.spx_2_gru = Conv2x(32, 32, deconv=True, bn=False, concat=True)
    self.spx_gru = nn.Sequential(
          nn.ConvTranspose2d(2*32, 9, kernel_size=4, stride=2, padding=1),
          )

    self.corr_stem = nn.Sequential(
      nn.Conv3d(self.proj_cmb.out_channels*2+self.cv_group, volume_dim, kernel_size=1),
      BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
      ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
      ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3, stride=1, padding=1),
    )
    self.corr_feature_att = FeatureAtt(volume_dim, self.feature.d_out[0])
    self.cost_agg = hourglass(cfg=self.args, in_channels=volume_dim, feat_dims=self.feature.d_out)
    self.classifier = nn.Sequential(
      BasicConv(volume_dim, volume_dim//2, kernel_size=3, padding=1, is_3d=True),
      ResnetBasicBlock3D(volume_dim//2, volume_dim//2, kernel_size=3, stride=1, padding=1),
      nn.Conv3d(volume_dim//2, 1, kernel_size=7, padding=3),
    )

    r = self.args.corr_radius
    dx = torch.arange(-r, r+1, requires_grad=False, dtype=torch.int8).reshape(1, 1, 2*r+1, 1)
    self.register_buffer("dx", dx)


  def upsample_disp(self, disp, mask_feat_4, stem_2x):
    with torch.amp.autocast('cuda', enabled=self.args.mixed_precision, dtype=U.AMP_DTYPE):
      xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
      spx_pred = self.spx_gru(xspx)
      spx_pred = F.softmax(spx_pred, 1)
      up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)
    return up_disp.to(self.dtype)


  def forward(self, image1, image2, iters=12, test_mode=False, low_memory=False, init_disp=None, profile=False, optimize_build_volume='pytorch1'):
    """ Estimate disparity between pair of frames """
    B,C,H,W = image1.shape
    low_memory = low_memory or (self.args.get('low_memory', False))
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)
    with torch.amp.autocast('cuda', enabled=self.args.mixed_precision, dtype=U.AMP_DTYPE):
      out = self.feature(torch.cat([image1, image2], dim=0))
      features_left = [o[:B] for o in out]
      features_right = [o[B:] for o in out]
      stem_2x = self.stem_2(image1)

      if optimize_build_volume=='pytorch1':
        gwc_volume = build_gwc_volume_optimized_pytorch1(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group, normalize=self.args.normalize)
      elif optimize_build_volume=='triton':
        gwc_volume = build_gwc_volume_triton(features_left[0], features_right[0], self.args.max_disp//4, self.cv_group, normalize=self.args.normalize)
      else:
        raise RuntimeError(f"Invalid optimize_build_volume: {optimize_build_volume}")

      left_tmp = self.proj_cmb(features_left[0])
      right_tmp = self.proj_cmb(features_right[0])
      concat_volume = build_concat_volume_optimized_pytorch1(left_tmp, right_tmp, maxdisp=self.args.max_disp//4)
      del left_tmp, right_tmp
      comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
      del concat_volume, gwc_volume

      comb_volume = self.corr_stem(comb_volume)
      comb_volume = self.corr_feature_att(comb_volume, features_left[0])
      comb_volume = self.cost_agg(comb_volume, features_left)

      # Init disp from geometry encoding volume
      logits = self.classifier(comb_volume).squeeze(1)
      prob = F.softmax(logits, dim=1)
      if init_disp is None:
        init_disp = disparity_regression(prob, self.args.max_disp//4)

      cnet_list = self.cnet(features_left[0], features_left[1], features_left[2])
      cnet_list = list(cnet_list)
      net_list = [torch.tanh(x[0]) for x in cnet_list]
      inp_list = [torch.relu(x[1]) for x in cnet_list]
      inp_list = [self.cam(x) * x for x in inp_list]
      att = [self.sam(x) for x in inp_list]

    geo_fn = Combined_Geo_Encoding_Volume(features_left[0].to(self.dtype), features_right[0].to(self.dtype), comb_volume.to(self.dtype), num_levels=self.args.corr_levels)
    b, c, h, w = features_left[0].shape
    coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
    disp = init_disp.to(self.dtype)
    disp_preds = []

    del comb_volume, features_left, features_right, cnet_list

    # GRUs iterations to update disparity (1/4 resolution)
    for itr in range(iters):
      disp = disp.detach()
      geo_feat = geo_fn(disp, coords, dx=self.dx, low_memory=low_memory)
      with torch.amp.autocast('cuda', enabled=self.args.mixed_precision, dtype=U.AMP_DTYPE):
        net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat.to(self.dtype), disp, att)

      disp = disp + delta_disp.to(self.dtype)
      if test_mode and itr < iters-1:
        continue

      # upsample predictions
      disp_up = self.upsample_disp(disp.to(self.dtype), mask_feat_4.to(self.dtype), stem_2x.to(self.dtype))
      disp_preds.append(disp_up)


    if test_mode:
      return disp_up

    return init_disp, disp_preds


  def run_hierachical(self, image1, image2, iters=12, test_mode=False, low_memory=False, small_ratio=0.5):
      B,_,H,W = image1.shape
      img1_small = F.interpolate(image1, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      img2_small = F.interpolate(image2, scale_factor=small_ratio, align_corners=False, mode='bilinear')
      padder = InputPadder(img1_small.shape[-2:], divis_by=32, force_square=False)
      img1_small, img2_small = padder.pad(img1_small, img2_small)
      disp_small = self.forward(img1_small, img2_small, test_mode=True, iters=iters, low_memory=low_memory)
      disp_small = padder.unpad(disp_small)
      disp_small_up = F.interpolate(disp_small, size=(H,W), mode='bilinear', align_corners=True) * 1/small_ratio
      disp_small_up = disp_small_up.clip(0, None)

      padder = InputPadder(image1.shape[-2:], divis_by=32, force_square=False)
      image1, image2, disp_small_up = padder.pad(image1, image2, disp_small_up)
      disp_small_up += padder._pad[0]
      init_disp = F.interpolate(disp_small_up, scale_factor=0.25, mode='bilinear', align_corners=True) * 0.25   # Init disp will be 1/4
      disp = self.forward(image1, image2, iters=iters, test_mode=test_mode, low_memory=low_memory, init_disp=init_disp)
      disp = padder.unpad(disp)
      return disp

FoundationStereoLite = FastFoundationStereo


class TrtFeatureRunner(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.feature = model.feature
    self.stem_2 = model.stem_2

  def forward(self, image1, image2):
    image1 = normalize_image(image1)
    image2 = normalize_image(image2)
    B = len(image1)
    out = self.feature(torch.cat([image1, image2], dim=0))
    features_left = [o[:B] for o in out]
    features_right = [o[B:] for o in out]
    stem_2x = self.stem_2(image1)
    return *features_left, features_right[0], stem_2x


class TrtPostRunner(nn.Module):
  def __init__(self, model):
    super().__init__()
    self.args = model.args
    self.dtype = model.dtype
    self.register_buffer("dx", model.dx)
    self.proj_cmb = model.proj_cmb
    self.corr_stem = model.corr_stem
    self.corr_feature_att = model.corr_feature_att
    self.cost_agg = model.cost_agg
    self.classifier = model.classifier
    self.update_block = model.update_block
    self.sam = model.sam
    self.cam = model.cam
    self.feature = model.feature
    self.spx_2_gru = model.spx_2_gru
    self.spx_gru = model.spx_gru
    self.cnet = model.cnet

  def upsample_disp(self, disp, mask_feat_4, stem_2x):
    with torch.amp.autocast('cuda', enabled=self.args.mixed_precision, dtype=U.AMP_DTYPE):
      xspx = self.spx_2_gru(mask_feat_4, stem_2x)   # 1/2 resolution
      spx_pred = self.spx_gru(xspx)
      spx_pred = F.softmax(spx_pred, 1)
      up_disp = context_upsample(disp*4., spx_pred).unsqueeze(1)
    return up_disp.to(self.dtype)


  def forward(self, features_left_04, features_left_08, features_left_16, features_left_32, features_right_04, stem_2x, gwc_volume):
    features_left = [features_left_04, features_left_08, features_left_16, features_left_32]
    left_tmp = self.proj_cmb(features_left_04)
    right_tmp = self.proj_cmb(features_right_04)
    concat_volume = build_concat_volume_optimized_pytorch(left_tmp, right_tmp, maxdisp=self.args.max_disp//4)
    del left_tmp, right_tmp
    comb_volume = torch.cat([gwc_volume, concat_volume], dim=1)
    del concat_volume, gwc_volume
    comb_volume = self.corr_stem(comb_volume)
    comb_volume = self.corr_feature_att(comb_volume, features_left_04)
    comb_volume = self.cost_agg(comb_volume, features_left)

    # Init disp from geometry encoding volume
    logits = self.classifier(comb_volume).squeeze(1)
    prob = F.softmax(logits, dim=1)
    init_disp = disparity_regression(prob, self.args.max_disp//4)

    cnet_list = self.cnet(features_left[0], features_left[1], features_left[2])
    cnet_list = list(cnet_list)
    net_list = [torch.tanh(x[0]) for x in cnet_list]
    inp_list = [torch.relu(x[1]) for x in cnet_list]
    inp_list = [self.cam(x) * x for x in inp_list]
    att = [self.sam(x) for x in inp_list]

    geo_fn = Combined_Geo_Encoding_Volume(features_left_04.to(self.dtype), features_right_04.to(self.dtype), comb_volume.to(self.dtype), num_levels=self.args.corr_levels)
    b, c, h, w = features_left[0].shape
    coords = torch.arange(w, dtype=torch.float, device=init_disp.device).reshape(1,1,w,1).repeat(b, h, 1, 1)
    disp = init_disp.to(self.dtype)

    # GRUs iterations to update disparity (1/4 resolution)
    for itr in range(self.args.valid_iters):
      disp = disp.detach()
      geo_feat = geo_fn(disp, coords, dx=self.dx, low_memory=True)
      net_list, mask_feat_4, delta_disp = self.update_block(net_list, inp_list, geo_feat.to(self.dtype), disp, att)

      disp = disp + delta_disp.to(self.dtype)
      if itr < self.args.valid_iters-1:
        continue

      disp_up = self.upsample_disp(disp.to(self.dtype), mask_feat_4.to(self.dtype), stem_2x.to(self.dtype))

    return disp_up


class TrtRunner(nn.Module):
  def __init__(self, args, feature_runner_engine_path, post_runner_engine_path):
    super().__init__()
    import tensorrt as trt
    self.args = args
    with open(feature_runner_engine_path, 'rb') as file:
      engine_data = file.read()
    self.trt_logger = trt.Logger(trt.Logger.WARNING)
    self.feature_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_data)
    self.feature_context = self.feature_engine.create_execution_context()

    with open(post_runner_engine_path, 'rb') as file:
      engine_data = file.read()
    self.post_engine = trt.Runtime(self.trt_logger).deserialize_cuda_engine(engine_data)
    self.post_context = self.post_engine.create_execution_context()
    self.max_disp = args.max_disp
    self.cv_group = args.get('cv_group', 8)

  def trt_dtype_to_torch(self, dt):
    import tensorrt as trt
    if dt==trt.DataType.FLOAT: return torch.float32
    if dt==trt.DataType.HALF: return torch.float16
    if dt==trt.DataType.BF16: return torch.bfloat16
    if dt==trt.DataType.INT32: return torch.int32
    if dt==trt.DataType.INT8: return torch.int8
    if dt==trt.DataType.BOOL: return torch.bool
    raise RuntimeError(f"Unsupported TRT dtype: {dt}")

  def get_io_tensor_names(self, engine, mode):
    names = []
    n = engine.num_io_tensors
    for i in range(n):
      name = engine.get_tensor_name(i)
      if engine.get_tensor_mode(name)==mode:
        names.append(name)
    return names

  def run_trt(self, engine, context, inputs_by_name:dict):
    import tensorrt as trt
    for name, tensor in list(inputs_by_name.items()):
      expected_dtype = self.trt_dtype_to_torch(engine.get_tensor_dtype(name))
      if tensor.dtype != expected_dtype: inputs_by_name[name] = tensor.to(expected_dtype)
      if not inputs_by_name[name].is_contiguous(): inputs_by_name[name] = inputs_by_name[name].contiguous()
      context.set_input_shape(name, tuple(inputs_by_name[name].shape))
    outputs = {}
    out_names = [n for n in self.get_io_tensor_names(engine, trt.TensorIOMode.OUTPUT)]
    for name in out_names:
      shp = tuple(context.get_tensor_shape(name))
      dtype = self.trt_dtype_to_torch(engine.get_tensor_dtype(name))
      outputs[name] = torch.empty(shp, device='cuda', dtype=dtype)
    for name, tensor in inputs_by_name.items(): context.set_tensor_address(name, int(tensor.data_ptr()))
    for name, tensor in outputs.items(): context.set_tensor_address(name, int(tensor.data_ptr()))
    stream = torch.cuda.current_stream().cuda_stream
    ok = context.execute_async_v3(stream)
    assert ok
    return outputs

  def forward(self, image1, image2):
    import tensorrt as trt
    feat_out = self.run_trt(self.feature_engine, self.feature_context, {'left': image1, 'right': image2})
    gwc_volume = build_gwc_volume_triton(feat_out['features_left_04'].half(), feat_out['features_right_04'].half(), self.args.max_disp//4, self.cv_group)
    post_inputs = feat_out
    post_inputs['gwc_volume'] = gwc_volume
    in_names = self.get_io_tensor_names(self.post_engine, trt.TensorIOMode.INPUT)
    tmp_keys = list(post_inputs.keys())
    for k in tmp_keys:
      if k not in in_names:
        del post_inputs[k]
    out = self.run_trt(self.post_engine, self.post_context, post_inputs)
    disp = out['disp']
    return disp


class OnnxRunner:
  """CPU-compatible inference runner using ONNX Runtime.

  Mirrors TrtRunner but uses onnxruntime.InferenceSession instead of TensorRT
  engines.  The GWC cost-volume computation that bridges the two models is
  handled by the pure-PyTorch fallback (build_gwc_volume_optimized_pytorch1),
  so no CUDA, Triton, or TensorRT installation is required.

  NOTE: The ONNX models exported by scripts/make_onnx.py have *fixed* input
  shapes (dynamic_axes are not set).  Passing images of a different size than
  was used during export will raise a shape-mismatch error from ORT.
  """

  def __init__(self, args, feature_onnx_path, post_onnx_path):
    try:
      import onnxruntime as ort
    except ImportError:
      raise ImportError(
          'onnxruntime is not installed. '
          'Install it with:  pip install onnxruntime'
      )
    self.args = args
    providers = ['CPUExecutionProvider']
    self.feature_session = ort.InferenceSession(feature_onnx_path, providers=providers)
    self.post_session    = ort.InferenceSession(post_onnx_path,    providers=providers)
    self.cv_group = args.get('cv_group', 8)

  @staticmethod
  def _to_np(x):
    """Convert a torch Tensor or array-like to a float32 numpy array."""
    if isinstance(x, torch.Tensor):
      return x.float().cpu().numpy()
    return np.asarray(x, dtype=np.float32)

  def forward(self, image1, image2):
    # --- Feature extraction ---
    feat_out_list = self.feature_session.run(
        None, {'left': self._to_np(image1), 'right': self._to_np(image2)}
    )
    feat_names = [o.name for o in self.feature_session.get_outputs()]
    feat = dict(zip(feat_names, feat_out_list))

    # --- GWC cost-volume (pure PyTorch, CPU-compatible) ---
    feat_l04 = torch.from_numpy(feat['features_left_04'])
    feat_r04 = torch.from_numpy(feat['features_right_04'])
    with torch.no_grad():
      gwc_volume = build_gwc_volume_optimized_pytorch1(
          feat_l04, feat_r04,
          self.args.max_disp // 4, self.cv_group,
          normalize=self.args.get('normalize', True),
      )

    # --- Post-processing ---
    post_inputs = dict(feat)
    post_inputs['gwc_volume'] = gwc_volume.numpy()
    # Cast each input to the dtype the post model actually expects
    _ort_to_np = {
        'tensor(float16)': np.float16,
        'tensor(float)':   np.float32,
        'tensor(double)':  np.float64,
    }
    post_inputs = {
        inp.name: post_inputs[inp.name].astype(_ort_to_np.get(inp.type, np.float32))
        for inp in self.post_session.get_inputs()
        if inp.name in post_inputs
    }

    out = self.post_session.run(None, post_inputs)
    disp = out[0]  # numpy array (B, 1, H, W)
    return disp