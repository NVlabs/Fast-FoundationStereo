import torch,os,sys
code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../../')
import torch.nn.functional as F
import numpy as np


class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, mode='sintel', divis_by=8, force_square=False):
        self.ht, self.wd = dims[-2:]
        if force_square:
          max_side = max(self.ht, self.wd)
          pad_ht = ((max_side // divis_by) + 1) * divis_by - self.ht
          pad_wd = ((max_side // divis_by) + 1) * divis_by - self.wd
        else:
          pad_ht = (((self.ht // divis_by) + 1) * divis_by - self.ht) % divis_by
          pad_wd = (((self.wd // divis_by) + 1) * divis_by - self.wd) % divis_by
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        assert all((x.ndim == 4) for x in inputs)
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self, x):
        assert x.ndim == 4
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]


@torch.compile
def bilinear_sampler1d(img, x_coords, mode='bilinear', align_corners=True):
    """
    1D bilinear sampling along width dimension only (for stereo applications)
    Much faster than grid_sample for stereo where y is constant

    Args:
        img: (B, C, 1, W) input tensor
        x_coords: (B, 1, W_out, 1) x coordinates in pixel space [0, W-1]
        mode: interpolation mode ('bilinear' or 'nearest')
        align_corners: if True, corner pixels are aligned (like grid_sample)

    Returns:
        sampled: (B, C, 1, W_coords) sampled tensor
        mask: (B, 1, H, W) validity mask (if mask=True)
    """
    B, C, H_img, W = img.shape
    x = x_coords.reshape(B,-1) # (B, W_out)

    if align_corners:
        # align_corners=True: coordinate range [0, W-1] maps to pixel centers
        # This matches grid_sample with align_corners=True behavior
        x_normalized = x
    else:
        # align_corners=False: coordinate range [0, W-1] maps to pixel edges
        # Need to adjust coordinates to match grid_sample with align_corners=False
        # grid_sample maps [-1, 1] to [0, W-1] when align_corners=False
        # So our [0, W-1] input should be treated as [0.5, W-0.5] in pixel space
        x_normalized = x + 0.5

    if mode == 'nearest':
        # Nearest neighbor sampling with zero padding outside [0, W-1]
        if align_corners:
            x_nearest = torch.round(x_normalized).long()
        else:
            x_nearest = torch.floor(x_normalized).long()
        valid = (x_nearest >= 0) & (x_nearest < W)  # (B, W_out)
        x_index = torch.clamp(x_nearest, 0, W-1)
        sampled = torch.gather(img, 3, x_index.view(B,1,1,-1).expand(B,C,1,-1))
        sampled = sampled * valid.view(B,1,1,-1).to(img.dtype)

    else:  # bilinear
        # Get integer and fractional parts
        x_floor = torch.floor(x_normalized)
        x_ceil = x_floor + 1
        x_frac = x_normalized - x_floor  # (B, W_out)

        # Zero padding behavior: mark validity and zero-out invalid contributions
        valid_floor = (x_floor >= 0) & (x_floor < W)
        valid_ceil = (x_ceil >= 0) & (x_ceil < W)
        x_floor_clamped = torch.clamp(x_floor, 0, W-1)
        x_ceil_clamped = torch.clamp(x_ceil, 0, W-1)

        # Create index tensors
        batch_idx = torch.arange(B, device=img.device).view(B, 1)
        img_floor = torch.gather(img, 3, x_floor_clamped.view(B,1,1,-1).expand(B,C,1,-1).long())
        img_ceil = torch.gather(img, 3, x_ceil_clamped.view(B,1,1,-1).expand(B,C,1,-1).long())

        # Apply validity masks (zero out-of-bounds samples)
        img_floor = img_floor * valid_floor.view(B,1,1,-1).to(img.dtype)
        img_ceil = img_ceil * valid_ceil.view(B,1,1,-1).to(img.dtype)

        # Linear interpolation
        x_frac = x_frac.view(B,1,1,-1)
        sampled = img_floor * (1 - x_frac) + img_ceil * x_frac

    return sampled


def bilinear_sampler(img, coords, mode='bilinear', mask=False, low_memory=False, use1d=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    H, W = img.shape[-2:]
    coords[...,0] = 2*coords[...,0]/(W-1) - 1
    if low_memory:
      B = img.shape[0]
      out = []
      bs = 102400
      for b in np.arange(0,B,bs):
        tmp = F.grid_sample(img[b:b+bs], coords[b:b+bs], align_corners=True)
        out.append(tmp)
      img = torch.cat(out, dim=0)
    else:
      img = F.grid_sample(img, coords, align_corners=True)
    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()
    return img

