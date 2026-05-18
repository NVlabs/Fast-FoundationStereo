import torch,os,sys
import torch.nn as nn
import torch.nn.functional as F
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from core.submodule import Conv2x_IN
import timm



class ContextNetSharedBackbone(nn.Module):
  def __init__(self, args, c04, c08, c16, output_dim=[(128,128,128), (128,128,128)], norm_fn='batch', downsample=3):
    super().__init__()
    self.args = args
    self.conv04 = nn.ModuleList([
      nn.Conv2d(c04, output_dim[0][0], kernel_size=3, padding=1),
      nn.Conv2d(c04, output_dim[1][0], kernel_size=3, padding=1),
    ])

  def forward(self, x4, x8, x16):
    outputs04 = []
    for i in range(len(self.conv04)):
      outputs04.append(self.conv04[i](x4))
    return (outputs04,)



class DepthAnythingFeature:
    model_configs = {
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }



class Feature(nn.Module):
    def __init__(self, args):
        super(Feature, self).__init__()
        self.args = args
        model = timm.create_model('edgenext_small', pretrained=True, features_only=False)
        self.stem = model.stem
        self.stages = model.stages
        chans = [48, 96, 160, 304]
        self.chans = chans
        vit_feat_dim = DepthAnythingFeature.model_configs[self.args.vit_size]['features']//2

        self.deconv32_16 = Conv2x_IN(chans[3], chans[2], deconv=True, concat=True)
        self.deconv16_8 = Conv2x_IN(chans[2]*2, chans[1], deconv=True, concat=True)
        self.deconv8_4 = Conv2x_IN(chans[1]*2, chans[0], deconv=True, concat=True)

        self.conv4 = nn.Conv2d(chans[0]*2, self.chans[0]*2+vit_feat_dim, kernel_size=1, stride=1, padding=0)

        self.d_out = [self.chans[0]*2+vit_feat_dim, self.chans[1]*2, self.chans[2]*2, self.chans[3]]


    def forward(self, x):
        B,C,H,W = x.shape
        if hasattr(self, 'stem'):
          x = self.stem(x)
          x4 = self.stages[0](x)
          x8 = self.stages[1](x4)
          x16 = self.stages[2](x8)
          x32 = self.stages[3](x16)
        else:
          intermediates = self.model.forward_intermediates(x, intermediates_only=True)
          x4, x8, x16, x32 = intermediates[-4:]

        with torch.profiler.record_function("feature_deconv"):
          x16 = self.deconv32_16(x32, x16)
          x8 = self.deconv16_8(x16, x8)
          x4 = self.deconv8_4(x8, x4)
          x4 = self.conv4(x4)
          if hasattr(self, 'conv8'):
            x8 = self.conv8(x8)
            x16 = self.conv16(x16)
            x32 = self.conv32(x32)
        return [x4, x8, x16, x32]
