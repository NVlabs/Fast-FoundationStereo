import os,sys
code_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(code_dir+'/../')
from foundation_stereo_ori.submodule import FeatureAtt
import torch
import torch.nn as nn
import Utils as U
import pickle


class ForwardHelper(nn.Module):
  def __init__(self, layers:list):
    super().__init__()
    self.layers = nn.ModuleList(layers)

  def forward(self, x, left_feat=None):
    for layer in self.layers:
      if isinstance(layer, FeatureAtt):
        x = layer(x, left_feat)
      else:
        x = layer(x)
    return x


class PostForwardHelper(nn.Module):
  def __init__(self, layers:list):
    super().__init__()
    for pos in range(len(layers)):
      if layers[pos] in ['sum', 'concat']:
        self.op = layers[pos]
        break
    self.upsample = nn.Sequential(*layers[:pos])
    self.out = nn.ModuleList(layers[pos+1:])

  def forward(self, conv2, conv3, left_feat=None):
    conv3_up = self.upsample(conv3)
    if self.op == 'sum':
      x = conv3_up + conv2
    elif self.op == 'concat':
      x = torch.cat((conv3_up, conv2), dim=1)
    else:
      raise ValueError(f"Unknown operation: {self.op}")

    for layer in self.out:
      if isinstance(layer, FeatureAtt):
        x = layer(x, left_feat)
      else:
        x = layer(x)
    return x

