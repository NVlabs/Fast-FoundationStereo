import torch,pdb,os,sys
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
from Utils import AMP_DTYPE
import Utils as U
try:
  import triton
  import triton.language as tl
except Exception:
  triton = None
  tl = None

def _is_contiguous(tensor: torch.Tensor) -> bool:
    if torch.jit.is_scripting():
        return tensor.is_contiguous()
    else:
        return tensor.is_contiguous(memory_format=torch.contiguous_format)



class LayerNorm2d(nn.LayerNorm):
    r""" https://huggingface.co/spaces/Roll20/pet_score/blob/b258ef28152ab0d5b377d9142a23346f863c1526/lib/timm/models/convnext.py#L85
    LayerNorm for channels_first tensors with 2d spatial dimensions (ie N, C, H, W).
    """

    def __init__(self, normalized_shape, eps=1e-6):
        """
        @normalized_shape: channel dim
        """
        super().__init__(normalized_shape, eps=eps)

    def forward(self, x) -> torch.Tensor:
        """
        @x: (B,C,H,W)
        """
        if _is_contiguous(x):
            return F.layer_norm(x.permute(0, 2, 3, 1), self.normalized_shape, self.weight, self.bias, self.eps).permute(0, 3, 1, 2).contiguous()
        else:
            s, u = torch.var_mean(x, dim=1, keepdim=True)
            x = (x - u) * torch.rsqrt(s + self.eps)
            x = x * self.weight[:, None, None] + self.bias[:, None, None]
            return x


class BasicConv(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, bn=True, relu=True, norm='batch', **kwargs):
        super(BasicConv, self).__init__()

        self.relu = nn.LeakyReLU(inplace=True) if relu else nn.Identity()
        self.use_bn = bn
        self.bn = nn.Identity()
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm3d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            if self.use_bn:
              if norm=='batch':
                self.bn = nn.BatchNorm2d(out_channels)
              elif norm=='instance':
                self.bn = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_bn:
            x = self.bn(x)
        if isinstance(self.relu, bool):
          if self.relu:
            self.relu = nn.LeakyReLU(inplace=True)
          else:
            self.relu = nn.Identity()
        x = self.relu(x)
        return x


class Conv3dNormActReduced(nn.Module):
    def __init__(self, C_in, C_out, hidden=None, kernel_size=3, kernel_disp=None, stride=1, norm=nn.BatchNorm3d):
        super().__init__()
        if kernel_disp is None:
          kernel_disp = kernel_size
        if hidden is None:
            hidden = C_out
        self.conv1 = nn.Sequential(
            nn.Conv3d(C_in, hidden, kernel_size=(1,kernel_size,kernel_size), padding=(0, kernel_size//2, kernel_size//2), stride=(1, stride, stride)),
            norm(hidden),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(hidden, C_out, kernel_size=(kernel_disp, 1, 1), padding=(kernel_disp//2, 0, 0), stride=(stride, 1, 1)),
            norm(C_out),
            nn.ReLU(),
        )


    def forward(self, x):
        """
        @x: (B,C,D,H,W)
        """
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class ResnetBasicBlock(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm2d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel_size, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out


class ResnetBasicBlock3D(nn.Module):
  def __init__(self, inplanes, planes, kernel_size=3, stride=1, padding=1, downsample=None, groups=1, base_width=64, dilation=1, norm_layer=nn.BatchNorm3d, bias=False):
    super().__init__()
    self.norm_layer = norm_layer
    if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
    if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
    # Both self.conv1 and self.downsample layers downsample the input when stride != 1
    self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=kernel_size, stride=stride, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn1 = norm_layer(planes)
    self.relu = nn.ReLU(inplace=True)
    self.conv2 = nn.Conv3d(planes, planes, kernel_size=kernel_size, bias=bias, padding=padding)
    if self.norm_layer is not None:
      self.bn2 = norm_layer(planes)
    self.downsample = downsample
    self.stride = stride


  def forward(self, x):
    identity = x

    out = self.conv1(x)
    if self.norm_layer is not None:
      out = self.bn1(out)
    out = self.relu(out)

    out = self.conv2(out)
    if self.norm_layer is not None:
      out = self.bn2(out)

    if self.downsample is not None:
      identity = self.downsample(x)
    out += identity
    out = self.relu(out)

    return out


class FlashMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, query, key, value, attn_mask=None, window_size=(-1,-1)):
        """
        @query: (B,L,C)
        """
        B,L,C = query.shape
        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(Q.size(0), Q.size(1), self.num_heads, self.head_dim)
        K = K.view(K.size(0), K.size(1), self.num_heads, self.head_dim)
        V = V.view(V.size(0), V.size(1), self.num_heads, self.head_dim)

        attn_output = F.scaled_dot_product_attention(Q, K, V)

        attn_output = attn_output.reshape(B,L,-1)
        output = self.out_proj(attn_output)

        return output



class FlashAttentionTransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_feedforward, dropout=0.1, act=nn.GELU, norm=nn.LayerNorm):
        super().__init__()
        self.self_attn = FlashMultiheadAttention(embed_dim, num_heads)
        self.act = act()

        self.linear1 = nn.Linear(embed_dim, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, embed_dim)

        self.norm1 = norm(embed_dim)
        self.norm2 = norm(embed_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, window_size=(-1, -1)):
        dtype = src.dtype
        src2 = self.self_attn(src, src, src, src_mask, window_size=window_size)
        src = src + self.dropout1(src2)
        src = self.norm1(src).to(dtype)

        src2 = self.linear2(self.dropout(self.act(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src).to(dtype)

        return src


class Conv2x(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, concat=True, keep_concat=True, bn=True, relu=True, keep_dispc=False):
        super(Conv2x, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv(in_channels, out_channels, deconv, is_3d, bn=bn, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = BasicConv(out_channels*2, out_channels*mul, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)
        else:
            self.conv2 = BasicConv(out_channels, out_channels, False, is_3d, bn, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x


class BasicConv_IN(nn.Module):

    def __init__(self, in_channels, out_channels, deconv=False, is_3d=False, IN=True, relu=True, **kwargs):
        super(BasicConv_IN, self).__init__()
        if relu:
          self.relu = nn.LeakyReLU(inplace=True)
        else:
          self.relu = nn.Identity()
        self.use_in = IN
        if is_3d:
            if deconv:
                self.conv = nn.ConvTranspose3d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv3d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm3d(out_channels)
        else:
            if deconv:
                self.conv = nn.ConvTranspose2d(in_channels, out_channels, bias=False, **kwargs)
            else:
                self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
            self.IN = nn.InstanceNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        if self.use_in:
          x = self.IN(x)
        if isinstance(self.relu, bool):
          if self.relu:
            self.relu = nn.LeakyReLU(inplace=True)
          else:
            self.relu = nn.Identity()
        x = self.relu(x)
        return x


class Conv2x_IN(nn.Module):
    def __init__(self, in_channels, out_channels, c_middle=None, deconv=False, is_3d=False, concat=True, keep_concat=True, IN=True, relu=True, keep_dispc=False):
        super(Conv2x_IN, self).__init__()
        self.concat = concat
        self.is_3d = is_3d
        if deconv and is_3d:
            kernel = (4, 4, 4)
        elif deconv:
            kernel = 4
        else:
            kernel = 3
        if c_middle is None:
          c_middle = out_channels

        if deconv and is_3d and keep_dispc:
            kernel = (1, 4, 4)
            stride = (1, 2, 2)
            padding = (0, 1, 1)
            self.conv1 = BasicConv_IN(in_channels, c_middle, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=stride, padding=padding)
        else:
            self.conv1 = BasicConv_IN(in_channels, c_middle, deconv, is_3d, IN=True, relu=True, kernel_size=kernel, stride=2, padding=1)

        if self.concat:
            mul = 2 if keep_concat else 1
            self.conv2 = ResnetBasicBlock(out_channels*2, out_channels*mul, kernel_size=3, stride=1, padding=1, norm_layer=nn.InstanceNorm2d)
        else:
            self.conv2 = BasicConv_IN(c_middle, out_channels, False, is_3d, IN, relu, kernel_size=3, stride=1, padding=1)

    def forward(self, x, rem):
        x = self.conv1(x)
        if x.shape != rem.shape:
            x = F.interpolate(x, size=(rem.shape[-2], rem.shape[-1]), mode='bilinear')
        if self.concat:
            x = torch.cat((x, rem), 1)
        else:
            x = x + rem
        x = self.conv2(x)
        return x



@torch.compile
def build_gwc_volume_optimized_pytorch1(refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int, num_groups: int, normalize=True):
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


if triton is not None and torch.cuda.is_available():
  @triton.autotune(configs=[
    triton.Config({'BLOCK_C':4,'BLOCK_W':128,'BLOCK_D':8}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_C':8,'BLOCK_W':128,'BLOCK_D':8}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_C':16,'BLOCK_W':128,'BLOCK_D':8}, num_warps=4, num_stages=2),
    triton.Config({'BLOCK_C':64,'BLOCK_W':128,'BLOCK_D':8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_C':128,'BLOCK_W':64,'BLOCK_D':8}, num_warps=8, num_stages=2),
    triton.Config({'BLOCK_C':128,'BLOCK_W':128,'BLOCK_D':8}, num_warps=8, num_stages=2),
  ], key=['C','W','D','G','K','NORMALIZE'])
  @triton.jit
  def _gwc_triton_kernel(ref_ptr, tar_ptr, ref_norm_ptr, tar_norm_ptr, out_ptr, BH, C, W, D: tl.constexpr, G: tl.constexpr, K: tl.constexpr,
                         stride_rn, stride_rw, stride_rc, stride_tn, stride_tw, stride_tc,
                         stride_nn, stride_ng, stride_nw,
                         stride_on, stride_og, stride_od, stride_ow,
                         NORMALIZE: tl.constexpr,
                         BLOCK_C: tl.constexpr, BLOCK_W: tl.constexpr, BLOCK_D: tl.constexpr):
    pid0 = tl.program_id(0)
    db = tl.program_id(1)
    wb = tl.program_id(2)
    bh = pid0 // G
    g = pid0 % G
    w_off = wb*BLOCK_W + tl.arange(0, BLOCK_W)
    d_off = db*BLOCK_D + tl.arange(0, BLOCK_D)
    w_mask = w_off < W
    w_src = w_off[None, :] - d_off[:, None]
    td_mask = (w_src >= 0) & w_mask[None, :]
    acc = tl.zeros((BLOCK_D, BLOCK_W), dtype=tl.float32)
    for k0 in tl.static_range(0, K, BLOCK_C):
      k_off = k0 + tl.arange(0, BLOCK_C)
      k_mask = k_off < K
      c_idx = g*K + k_off
      ref_ptrs = ref_ptr + bh*stride_rn + w_off[None, :]*stride_rw + c_idx[:, None]*stride_rc
      ref_vals = tl.load(ref_ptrs, mask=k_mask[:, None] & w_mask[None, :], other=0.).to(tl.float32)
      tar_ptrs = tar_ptr + bh*stride_tn + w_src[None, :, :]*stride_tw + c_idx[:, None, None]*stride_tc
      tar_vals = tl.load(tar_ptrs, mask=k_mask[:, None, None] & td_mask[None, :, :], other=0.).to(tl.float32)
      acc += tl.sum(tar_vals * ref_vals[:, None, :], axis=0)

    if NORMALIZE:
      norm_offset = bh*stride_nn + g*stride_ng
      ref_norm = tl.load(ref_norm_ptr + norm_offset + w_off*stride_nw, mask=w_mask, other=1.0).to(tl.float32)
      tar_norm = tl.load(tar_norm_ptr + norm_offset + w_src*stride_nw, mask=td_mask, other=1.0).to(tl.float32)
      denom = (ref_norm[None, :] * tar_norm) + 1e-5
      acc = acc / denom
    out_ptrs = out_ptr + bh*stride_on + g*stride_og + d_off[:, None]*stride_od + w_off[None, :]*stride_ow
    tl.store(out_ptrs, acc, mask=w_mask[None, :])

@torch.no_grad()
def build_gwc_volume_triton(refimg_fea: torch.Tensor, targetimg_fea: torch.Tensor, maxdisp: int, num_groups: int, normalize=True):
  if triton is None:
    raise RuntimeError('Triton is not available. Please install triton to use build_gwc_volume_triton.')
  B, C, H, W = refimg_fea.shape
  assert maxdisp > 0 and C % num_groups == 0
  K = C // num_groups
  in_dtype = refimg_fea.dtype if refimg_fea.dtype in (torch.float16, torch.bfloat16, torch.float32) else torch.float32

  if normalize:
    ref_norm = refimg_fea.float().view(B, num_groups, K, H, W).norm(dim=2)
    tar_norm = targetimg_fea.float().view(B, num_groups, K, H, W).norm(dim=2)
    ref_norm = ref_norm.permute(0, 2, 1, 3).reshape(B*H, num_groups, W).to(in_dtype).contiguous()
    tar_norm = tar_norm.permute(0, 2, 1, 3).reshape(B*H, num_groups, W).to(in_dtype).contiguous()
  else:
    # Dummy tensors; kernel won't read them when NORMALIZE=False
    ref_norm = refimg_fea.new_empty((1, 1, 1), dtype=in_dtype)
    tar_norm = refimg_fea.new_empty((1, 1, 1), dtype=in_dtype)

  ref = refimg_fea.to(in_dtype)
  tar = targetimg_fea.to(in_dtype)
  ref_bhwc = ref.permute(0, 2, 3, 1).view(B * H, W, C).contiguous()
  tar_bhwc = tar.permute(0, 2, 3, 1).view(B * H, W, C).contiguous()
  out_bhw = torch.empty((B * H, num_groups, maxdisp, W), device=ref.device, dtype=in_dtype)
  BH = B * H
  D_eff = min(maxdisp, W)
  grid = lambda META: (BH * num_groups, triton.cdiv(D_eff, META['BLOCK_D']), triton.cdiv(W, META['BLOCK_W']))
  _gwc_triton_kernel[grid](ref_bhwc, tar_bhwc, ref_norm, tar_norm, out_bhw, BH, C, W, D_eff, num_groups, K,
                           ref_bhwc.stride(0), ref_bhwc.stride(1), ref_bhwc.stride(2),
                           tar_bhwc.stride(0), tar_bhwc.stride(1), tar_bhwc.stride(2),
                           ref_norm.stride(0), ref_norm.stride(1), ref_norm.stride(2),
                           out_bhw.stride(0), out_bhw.stride(1), out_bhw.stride(2), out_bhw.stride(3),
                           NORMALIZE=normalize)
  if D_eff < maxdisp: out_bhw[:, :, D_eff:, :] = 0
  volume = out_bhw.view(B, H, num_groups, maxdisp, W).permute(0, 2, 3, 1, 4).contiguous()
  return volume



@torch.compile
def build_concat_volume_optimized_pytorch(refimg_fea, targetimg_fea, maxdisp:int):
  B, C, H, W = refimg_fea.shape
  ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
  shifted_target_list = [F.pad(targetimg_fea, (int(d), 0, 0, 0), "constant", 0.0)[:, :, :, :W] for d in range(maxdisp)]
  target_volume = torch.stack(shifted_target_list, dim=2)
  volume = torch.cat((ref_volume, target_volume), dim=1)
  return volume.contiguous()


@torch.compile
def build_concat_volume_optimized_pytorch1(refimg_fea, targetimg_fea, maxdisp:int):
    B, C, H, W = refimg_fea.shape

    ref_volume = refimg_fea.unsqueeze(2).expand(B, C, maxdisp, H, W)
    padded_target = F.pad(targetimg_fea, (maxdisp - 1, 0, 0, 0))  # (B, C, H, W + maxdisp - 1)
    unfolded_target = padded_target.unfold(dimension=3, size=W, step=1)  # (B, C, H, maxdisp, W)
    target_volume = torch.flip(unfolded_target, [3]).permute(0, 1, 3, 2, 4)
    volume = torch.cat((ref_volume, target_volume), dim=1)
    return volume.contiguous()




def disparity_regression(x, maxdisp):
    assert len(x.shape) == 4
    disp_values = torch.arange(0, maxdisp, dtype=x.dtype, device=x.device)
    disp_values = disp_values.reshape(1, maxdisp, 1, 1)
    return torch.sum(x * disp_values, 1, keepdim=True)   #(B,1,H,W)


class FeatureAtt(nn.Module):
    def __init__(self, cv_chan, feat_chan):
        super(FeatureAtt, self).__init__()

        self.feat_att = nn.Sequential(
            BasicConv(feat_chan, feat_chan//2, kernel_size=1, stride=1, padding=0),
            nn.Conv2d(feat_chan//2, cv_chan, 1)
            )

    def forward(self, cv, feat):
        '''
        @cv: cost volume (B,C,D,H,W)
        @feat: (B,C,H,W)
        '''
        feat_att = self.feat_att(feat).unsqueeze(2)   #(B,C,1,H,W)
        cv = torch.sigmoid(feat_att)*cv
        return cv

def context_upsample(disp_low, up_weights):
    """
    @disp_low: (b,1,h,w)  1/4 resolution
    @up_weights: (b,9,4*h,4*w)  Image resolution
    """
    b, c, h, w = disp_low.shape

    disp_unfold = F.unfold(disp_low.reshape(b,c,h,w),3,1,1).reshape(b,-1,h,w)
    disp_unfold = F.interpolate(disp_unfold,(h*4,w*4),mode='nearest').reshape(b,9,h*4,w*4)

    disp = (disp_unfold*up_weights).sum(1)

    return disp



class PositionalEmbedding(nn.Module):
  def __init__(self, d_model, max_len=512):
    super().__init__()

    # Compute the positional encodings once in log space.
    pe = torch.zeros(max_len, d_model, dtype=torch.float)
    pe.require_grad = False

    position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  #(N,1)
    div_term = (torch.arange(0, d_model, 2, dtype=torch.float) * -(np.log(10000.0) / d_model)).exp()[None]

    pe[:, 0::2] = torch.sin(position * div_term)  #(N, d_model/2)
    pe[:, 1::2] = torch.cos(position * div_term)

    pe = pe.unsqueeze(0)
    self.pe = pe


  def forward(self, x, resize_embed=False):
    '''
    @x: (B,N,D)
    '''
    dtype = x.dtype
    self.pe = self.pe.to(x.device).to(x.dtype)
    pe = self.pe
    if pe.shape[1]<x.shape[1]:
      if resize_embed:
        pe = F.interpolate(pe.permute(0,2,1), size=x.shape[1], mode='linear', align_corners=True).permute(0,2,1)
      else:
        raise RuntimeError(f'x:{x.shape}, pe:{pe.shape}')
    return (x + pe[:, :x.size(1)]).to(dtype)



class CostVolumeDisparityAttention(nn.Module):
  def __init__(self, d_model, nhead, dim_feedforward, dropout=0.1, act=nn.GELU, norm_first=False, num_transformer=6, max_len=512, resize_embed=False):
    super().__init__()
    self.resize_embed = resize_embed
    self.sa = nn.ModuleList([])
    for _ in range(num_transformer):
      self.sa.append(FlashAttentionTransformerEncoderLayer(embed_dim=d_model, num_heads=nhead, dim_feedforward=dim_feedforward, act=act, dropout=dropout))
    self.pos_embed0 = PositionalEmbedding(d_model, max_len=max_len)


  def forward(self, cv, window_size=(-1,-1)):
    """
    @cv: (B,C,D,H,W) where D is max disparity
    """
    x = cv
    B,C,D,H,W = x.shape
    x = x.permute(0,3,4,2,1).reshape(B*H*W, D, C)
    x = self.pos_embed0(x, resize_embed=self.resize_embed)  #!NOTE No resize since disparity is pre-determined
    for i in range(len(self.sa)):
        x = self.sa[i](x, window_size=window_size)
    x = x.reshape(B,H,W,D,C).permute(0,4,3,1,2)

    return x



class ChannelAttentionEnhancement(nn.Module):
    def __init__(self, in_planes, ratio=16):
        """From selective-IGEV
        """
        super(ChannelAttentionEnhancement, self).__init__()
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 16, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 16, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(torch.mean(x, dim=(2, 3), keepdim=True))
        max_out = self.fc(torch.amax(x, dim=(2, 3), keepdim=True))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttentionExtractor(nn.Module):
    def __init__(self, kernel_size=7):
        """From selective-IGEV
        """
        super(SpatialAttentionExtractor, self).__init__()

        self.samconv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.samconv(x)
        return self.sigmoid(x)



class EdgeNextConvEncoder(nn.Module):
    def __init__(self, dim, layer_scale_init_value=1e-6, expan_ratio=4, kernel_size=7, norm='layer'):
        """https://github.com/mmaaz60/EdgeNeXt/blob/main/models/conv_encoder.py#L7
        """
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim)
        if norm=='layer':
          self.norm = LayerNorm2d(dim, eps=1e-6)
        elif norm=='batch':
          self.norm = nn.BatchNorm2d(dim)
        else:
          self.norm = nn.Identity()
        self.pwconv1 = nn.Linear(dim, expan_ratio * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(expan_ratio * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + x
        return x
