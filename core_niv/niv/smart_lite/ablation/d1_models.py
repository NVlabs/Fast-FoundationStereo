"""
Direction-1 ablation models: SmartLite-L base, one FFS component swapped in.

Each class replaces exactly one sub-system with its FFS counterpart so we
can measure the per-component accuracy delta.

Variants
--------
D1_NoRsPrior   – eval only; zeros rs_disp / conf  (baseline minus prior)
D1_SPX         – replaces ConvexUpsample with FFS SPX upsample
D1_SelGRU      – replaces DS-ConvGRU with FFS SelectiveConvGRU
D1_RichVolume  – replaces tiny CoarseDisparityHead with combined GWC+concat
                 volume + 3-D stem + classifier (no hourglass, no multi-scale
                 features needed)
D1_EdgeNeXt    – replaces DS-conv backbone with FFS EdgeNeXt Feature extractor
                 + 1×1 adapter conv

Usage
-----
    from smart_lite.ablation.d1_models import build_d1_model
    model = build_d1_model('spx', max_disp=192)
    model = build_d1_model('edgenext', max_disp=192,
                           ffs_ckpt='../../weights/model_best_bp2.pth')
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.realpath(__file__))
_FFS_ROOT = os.path.abspath(os.path.join(_HERE, '..', '..'))
if _FFS_ROOT not in sys.path:
    sys.path.insert(0, _FFS_ROOT)

from core_niv.niv.smart_lite.model import (
    SmartLiteStereo, build_gwc_volume, MODEL_CONFIGS,
    DepthwiseSeparableConv2d, DepthwiseSeparableConvGRU,
    MotionEncoder,
)
from core.submodule import (
    Conv2x,
    context_upsample,
    build_gwc_volume_optimized_pytorch1,
    build_concat_volume_optimized_pytorch1,
    disparity_regression,
    BasicConv_IN,
)
from core.update import SelectiveConvGRU


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _L_cfg():
    """SmartLite-L hyperparameters."""
    return MODEL_CONFIGS['L']


def _load_ffs(ffs_ckpt: str, device='cpu'):
    """Load FastFoundationStereo, return frozen on *device*.

    Supports two checkpoint formats:
      - model_best_bp2_serialize.pth : serialized FastFoundationStereo object
        (pure EdgeNeXt, no DINOv2 — preferred)
      - model_best_bp2.pth : state-dict checkpoint (contains DINOv2 + EdgeNeXt;
        shape-filters to extract only EdgeNeXt feature.* keys)
    """
    obj = torch.load(ffs_ckpt, map_location='cpu', weights_only=False)

    if hasattr(obj, 'state_dict'):
        # Serialized FastFoundationStereo — pure EdgeNeXt, no DINOv2
        ffs = obj
        print(f'[_load_ffs] loaded serialized FastFoundationStereo from {ffs_ckpt}')
    else:
        # State-dict checkpoint: build model from cfg.yaml then load weights
        from omegaconf import OmegaConf
        cfg_path = os.path.join(os.path.dirname(ffs_ckpt), 'cfg.yaml')
        cfg  = OmegaConf.load(cfg_path)
        state = obj.get('model', obj.get('state_dict', obj))
        from core.foundation_stereo import FastFoundationStereo
        ffs  = FastFoundationStereo(cfg)
        model_state = ffs.state_dict()
        compatible = {k: v for k, v in state.items()
                      if k in model_state and v.shape == model_state[k].shape}
        skipped = len(state) - len(compatible)
        if skipped:
            print(f'[_load_ffs] skipped {skipped} shape-mismatched keys '
                  f'(DINOv2/dim mismatches — harmless, only feature.* extracted)')
        missing, unexpected = ffs.load_state_dict(compatible, strict=False)
        feature_missing = [k for k in missing if k.startswith('feature.')]
        if feature_missing:
            print(f'[_load_ffs] WARNING: feature keys missing: {feature_missing[:5]}')

    ffs = ffs.to(device).eval()
    for p in ffs.parameters():
        p.requires_grad = False
    cfg = getattr(ffs, 'cfg', None)
    return ffs, cfg


# ---------------------------------------------------------------------------
# D1_NoRsPrior — eval only, zeros the sensor prior
# ---------------------------------------------------------------------------

class D1_NoRsPrior(SmartLiteStereo):
    """SmartLite-L with RS prior disabled. No training changes needed."""

    def forward(self, left_rgb, right_rgb, rs_disp, conf,
                num_iters=None, test_mode=False):
        return super().forward(
            left_rgb, right_rgb,
            torch.zeros_like(rs_disp),
            torch.zeros_like(conf),
            num_iters=num_iters, test_mode=test_mode,
        )


# ---------------------------------------------------------------------------
# D1_SPX — FFS SPX upsample instead of ConvexUpsample
# ---------------------------------------------------------------------------

class D1_SPX(SmartLiteStereo):
    """SmartLite-L with FFS SPX upsampling.

    New modules (need training): stem_2, spx_2_gru, spx_gru
    Removed: upsample (ConvexUpsample)

    SPX uses the original-resolution image (1/2-res stem features) to sharpen
    depth-discontinuity boundaries, which ConvexUpsample cannot.
    """

    def __init__(self, max_disp=192, **sl_kwargs):
        super().__init__(max_disp=max_disp, **sl_kwargs)
        hidden_dim = sl_kwargs.get('hidden_dim', _L_cfg()['hidden_dim'])
        del self.upsample  # replaced

        # SPX modules matching FFS architecture
        self.stem_2 = nn.Sequential(
            BasicConv_IN(3, 32, kernel_size=3, stride=2, padding=1),
            nn.Conv2d(32, 32, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(32), nn.ReLU(),
        )
        self.spx_2_gru = Conv2x(hidden_dim, 32, deconv=True, bn=False, concat=True)
        self.spx_gru   = nn.ConvTranspose2d(2 * 32, 9, kernel_size=4,
                                            stride=2, padding=1)

    def _upsample(self, net, disp_low, stem_2x):
        xspx     = self.spx_2_gru(net, stem_2x)
        spx_pred = F.softmax(self.spx_gru(xspx), dim=1)
        return context_upsample(disp_low * 4., spx_pred).unsqueeze(1)

    def forward(self, left_rgb, right_rgb, rs_disp, conf,
                num_iters=None, test_mode=False):
        iters = num_iters if num_iters is not None else self.num_iters
        B, _, H, W = left_rgb.shape
        Hq, Wq = H // 4, W // 4

        stem_2x = self.stem_2(left_rgb)

        feat_left, feat_right = self.backbone(left_rgb, right_rgb)
        gwc_volume = build_gwc_volume(feat_left, feat_right,
                                      self.max_disp_q4, self.num_groups)

        rs_q4   = F.interpolate(rs_disp, (Hq, Wq), mode='bilinear',
                                align_corners=False) * 0.25
        conf_q4 = F.interpolate(conf,    (Hq, Wq), mode='bilinear',
                                align_corners=False)
        prior   = torch.cat([rs_q4, conf_q4], dim=1)

        coarse_disp = self.coarse_head(gwc_volume)
        has_sensor  = (conf_q4 > 0).float()
        disp = has_sensor * rs_q4 + (1.0 - has_sensor) * coarse_disp

        net, inp, att = self.context_net(feat_left)

        disp_preds = []
        for i in range(iters):
            disp  = disp.detach()
            corr  = self.corr_lookup(gwc_volume, disp)
            motion = self.motion_encoder(disp, corr, prior)
            net   = self.gru(net, torch.cat([motion, inp * att], dim=1))
            disp  = disp + self.disp_head(net)
            if test_mode and i < iters - 1:
                continue
            disp_preds.append(self._upsample(net, disp, stem_2x))

        return disp_preds[-1] if test_mode else (coarse_disp, disp_preds)


# ---------------------------------------------------------------------------
# D1_SelGRU — FFS SelectiveConvGRU cell, keep MotionEncoder + prior
# ---------------------------------------------------------------------------

class _GRUAttProj(nn.Module):
    """Predict per-pixel kernel-selection attention from GRU hidden state."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim // 2, 1, 1),
            nn.Sigmoid(),
        )
    def forward(self, h):
        return self.proj(h)


class D1_SelGRU(SmartLiteStereo):
    """SmartLite-L with FFS SelectiveConvGRU replacing DS-ConvGRU.

    New modules (need training): sel_gru, gru_att_proj
    Removed: gru (DepthwiseSeparableConvGRU)

    SelectiveConvGRU blends a 1×1 (fast, global) and 3×3 (spatial) GRU cell
    based on per-pixel attention — more expressive than a fixed DS-GRU.
    """

    def __init__(self, max_disp=192, **sl_kwargs):
        super().__init__(max_disp=max_disp, **sl_kwargs)
        cfg = _L_cfg()
        hidden_dim  = sl_kwargs.get('hidden_dim',  cfg['hidden_dim'])
        context_dim = sl_kwargs.get('context_dim', cfg['context_dim'])
        input_dim   = hidden_dim + context_dim   # motion + inp*att
        del self.gru  # replaced

        self.sel_gru      = SelectiveConvGRU(hidden_dim=hidden_dim,
                                             input_dim=input_dim)
        self.gru_att_proj = _GRUAttProj(hidden_dim)

    def forward(self, left_rgb, right_rgb, rs_disp, conf,
                num_iters=None, test_mode=False):
        iters = num_iters if num_iters is not None else self.num_iters
        B, _, H, W = left_rgb.shape
        Hq, Wq = H // 4, W // 4

        feat_left, feat_right = self.backbone(left_rgb, right_rgb)
        gwc_volume = build_gwc_volume(feat_left, feat_right,
                                      self.max_disp_q4, self.num_groups)

        rs_q4   = F.interpolate(rs_disp, (Hq, Wq), mode='bilinear',
                                align_corners=False) * 0.25
        conf_q4 = F.interpolate(conf,    (Hq, Wq), mode='bilinear',
                                align_corners=False)
        prior   = torch.cat([rs_q4, conf_q4], dim=1)

        coarse_disp = self.coarse_head(gwc_volume)
        has_sensor  = (conf_q4 > 0).float()
        disp = has_sensor * rs_q4 + (1.0 - has_sensor) * coarse_disp

        net, inp, att = self.context_net(feat_left)

        disp_preds = []
        for i in range(iters):
            disp   = disp.detach()
            corr   = self.corr_lookup(gwc_volume, disp)
            motion = self.motion_encoder(disp, corr, prior)
            x_in   = torch.cat([motion, inp * att], dim=1)
            gru_att = self.gru_att_proj(net)           # [B,1,Hq,Wq]
            net    = self.sel_gru(gru_att, net, x_in)
            disp   = disp + self.disp_head(net)
            if test_mode and i < iters - 1:
                continue
            disp_preds.append(self.upsample(net, disp))

        return disp_preds[-1] if test_mode else (coarse_disp, disp_preds)


# ---------------------------------------------------------------------------
# D1_RichVolume — combined GWC+concat volume + 3-D stem + classifier
#                 (FFS cost-agg init without the hourglass)
# ---------------------------------------------------------------------------

class D1_RichVolume(SmartLiteStereo):
    """SmartLite-L with a richer coarse init from GWC+concat+3D-stem.

    New modules (need training): proj_cmb, corr_stem, ffs_classifier
    Removed: coarse_head

    Note: hourglass and FeatureAtt are skipped because they require multi-scale
    features that SmartLite backbone doesn't produce.  This still tests whether
    a richer (combined) cost volume gives a better warm start over the tiny 2-
    layer 3D-conv head.
    """

    _CONCAT_HALF = 12    # FFS concat_channel // 2

    def __init__(self, max_disp=192, **sl_kwargs):
        super().__init__(max_disp=max_disp, **sl_kwargs)
        feature_ch = sl_kwargs.get('feature_ch', _L_cfg()['feature_ch'])
        num_groups  = sl_kwargs.get('num_groups', _L_cfg()['num_groups'])
        del self.coarse_head  # replaced

        volume_dim = 28   # FFS default
        ch_half    = self._CONCAT_HALF

        self.proj_cmb = nn.Conv2d(feature_ch, ch_half, 1, bias=False)

        from core.submodule import BasicConv, ResnetBasicBlock3D
        self.corr_stem = nn.Sequential(
            nn.Conv3d(ch_half * 2 + num_groups, volume_dim, kernel_size=1),
            BasicConv(volume_dim, volume_dim, kernel_size=3, padding=1, is_3d=True),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3,
                               stride=1, padding=1),
            ResnetBasicBlock3D(volume_dim, volume_dim, kernel_size=3,
                               stride=1, padding=1),
        )
        from core.submodule import ResnetBasicBlock3D as R3D
        self.ffs_classifier = nn.Sequential(
            BasicConv(volume_dim, volume_dim // 2, kernel_size=3,
                      padding=1, is_3d=True),
            R3D(volume_dim // 2, volume_dim // 2, kernel_size=3,
                stride=1, padding=1),
            nn.Conv3d(volume_dim // 2, 1, kernel_size=7, padding=3),
        )

    def _coarse_init(self, feat_left, feat_right):
        gwc = build_gwc_volume_optimized_pytorch1(
            feat_left, feat_right, self.max_disp_q4,
            self.num_groups, normalize=True)
        lc = self.proj_cmb(feat_left)
        rc = self.proj_cmb(feat_right)
        cat_vol = build_concat_volume_optimized_pytorch1(
            lc, rc, maxdisp=self.max_disp_q4)
        comb = self.corr_stem(torch.cat([gwc, cat_vol], dim=1))
        logits = self.ffs_classifier(comb).squeeze(1)
        return disparity_regression(F.softmax(logits, dim=1), self.max_disp_q4)

    def forward(self, left_rgb, right_rgb, rs_disp, conf,
                num_iters=None, test_mode=False):
        iters = num_iters if num_iters is not None else self.num_iters
        B, _, H, W = left_rgb.shape
        Hq, Wq = H // 4, W // 4

        feat_left, feat_right = self.backbone(left_rgb, right_rgb)
        gwc_volume = build_gwc_volume(feat_left, feat_right,
                                      self.max_disp_q4, self.num_groups)

        rs_q4   = F.interpolate(rs_disp, (Hq, Wq), mode='bilinear',
                                align_corners=False) * 0.25
        conf_q4 = F.interpolate(conf,    (Hq, Wq), mode='bilinear',
                                align_corners=False)
        prior   = torch.cat([rs_q4, conf_q4], dim=1)

        coarse_disp = self._coarse_init(feat_left, feat_right)
        has_sensor  = (conf_q4 > 0).float()
        disp = has_sensor * rs_q4 + (1.0 - has_sensor) * coarse_disp

        net, inp, att = self.context_net(feat_left)

        disp_preds = []
        for i in range(iters):
            disp   = disp.detach()
            corr   = self.corr_lookup(gwc_volume, disp)
            motion = self.motion_encoder(disp, corr, prior)
            net    = self.gru(net, torch.cat([motion, inp * att], dim=1))
            disp   = disp + self.disp_head(net)
            if test_mode and i < iters - 1:
                continue
            disp_preds.append(self.upsample(net, disp))

        return disp_preds[-1] if test_mode else (coarse_disp, disp_preds)


# ---------------------------------------------------------------------------
# D1_EdgeNeXt — FFS EdgeNeXt backbone + 1×1 adapter, keep everything else
# ---------------------------------------------------------------------------

class D1_EdgeNeXt(SmartLiteStereo):
    """SmartLite-L with FFS EdgeNeXt Feature extractor replacing DS backbone.

    New modules (need training): feat_adapter (1×1 conv)
    Frozen (if ffs_ckpt provided): ffs_feature (pretrained ImageNet weights)
    Removed: backbone

    The EdgeNeXt backbone is pretrained on ImageNet and outputs 128-160ch
    features at 1/4 resolution.  A 1×1 conv adapts this to SmartLite's
    expected feature_ch=64, so the downstream GWC / context_net / GRU
    continue to operate unchanged.
    """

    def __init__(self, ffs_ckpt: str = None, max_disp=192, **sl_kwargs):
        super().__init__(max_disp=max_disp, **sl_kwargs)
        feature_ch = sl_kwargs.get('feature_ch', _L_cfg()['feature_ch'])
        del self.backbone

        from core.extractor import Feature as FFSFeature
        if ffs_ckpt is not None:
            ffs_model, ffs_cfg = _load_ffs(ffs_ckpt)
            self.ffs_feature = ffs_model.feature
        else:
            from omegaconf import OmegaConf
            # Minimal stub args if no checkpoint provided
            ffs_cfg = OmegaConf.create({
                'vit_size': 'vits', 'mixed_precision': False,
            })
            self.ffs_feature = FFSFeature(ffs_cfg)

        ffs_x4_ch = self.ffs_feature.d_out[0]
        self.feat_adapter = nn.Conv2d(ffs_x4_ch, feature_ch, 1, bias=False)

    def _extract(self, left_rgb, right_rgb):
        imgs = torch.cat([left_rgb * 255.0, right_rgb * 255.0], dim=0)
        out  = self.ffs_feature(imgs)
        B    = left_rgb.shape[0]
        fl   = self.feat_adapter(out[0][:B])
        fr   = self.feat_adapter(out[0][B:])
        return fl.contiguous(), fr.contiguous()

    def forward(self, left_rgb, right_rgb, rs_disp, conf,
                num_iters=None, test_mode=False):
        iters = num_iters if num_iters is not None else self.num_iters
        B, _, H, W = left_rgb.shape
        Hq, Wq = H // 4, W // 4

        feat_left, feat_right = self._extract(left_rgb, right_rgb)
        gwc_volume = build_gwc_volume(feat_left, feat_right,
                                      self.max_disp_q4, self.num_groups)

        rs_q4   = F.interpolate(rs_disp, (Hq, Wq), mode='bilinear',
                                align_corners=False) * 0.25
        conf_q4 = F.interpolate(conf,    (Hq, Wq), mode='bilinear',
                                align_corners=False)
        prior   = torch.cat([rs_q4, conf_q4], dim=1)

        coarse_disp = self.coarse_head(gwc_volume)
        has_sensor  = (conf_q4 > 0).float()
        disp = has_sensor * rs_q4 + (1.0 - has_sensor) * coarse_disp

        net, inp, att = self.context_net(feat_left)

        disp_preds = []
        for i in range(iters):
            disp   = disp.detach()
            corr   = self.corr_lookup(gwc_volume, disp)
            motion = self.motion_encoder(disp, corr, prior)
            net    = self.gru(net, torch.cat([motion, inp * att], dim=1))
            disp   = disp + self.disp_head(net)
            if test_mode and i < iters - 1:
                continue
            disp_preds.append(self.upsample(net, disp))

        return disp_preds[-1] if test_mode else (coarse_disp, disp_preds)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_D1_CLASSES = {
    'no_prior':   D1_NoRsPrior,
    'spx':        D1_SPX,
    'selgru':     D1_SelGRU,
    'rich_volume': D1_RichVolume,
    'edgenext':   D1_EdgeNeXt,
}


def build_d1_model(variant: str, max_disp: int = 192,
                   ffs_ckpt: str = None,
                   num_iters: int = None,
                   feature_ch: int = None) -> SmartLiteStereo:
    """Build a Direction-1 hybrid model with L config.

    Args:
        variant:    one of no_prior | spx | selgru | rich_volume | edgenext
        max_disp:   maximum disparity (default 192)
        ffs_ckpt:   path to FFS checkpoint (required for edgenext)
        num_iters:  GRU iterations override (default: L config = 8).
        feature_ch: override feature width (default: L config = 64).
                    Use 32 for a lighter model.
    """
    if variant not in _D1_CLASSES:
        raise ValueError(f"Unknown D1 variant '{variant}'. "
                         f"Choose: {list(_D1_CLASSES)}")
    cfg = _L_cfg()
    kwargs = dict(
        feature_ch=feature_ch if feature_ch is not None else cfg['feature_ch'],
        backbone_mid1=cfg['backbone_mid1'],
        backbone_mid2=cfg['backbone_mid2'],
        backbone_extra_blocks=cfg['backbone_extra_blocks'],
        hidden_dim=cfg['hidden_dim'],
        context_dim=cfg['context_dim'],
        context_mid=cfg['context_mid'],
        motion_ch=cfg['motion_ch'],
        num_groups=cfg['num_groups'],
        corr_radius=cfg['corr_radius'],
        num_iters=num_iters if num_iters is not None else cfg['num_iters'],
    )
    cls = _D1_CLASSES[variant]
    if variant == 'edgenext':
        return cls(ffs_ckpt=ffs_ckpt, max_disp=max_disp, **kwargs)
    return cls(max_disp=max_disp, **kwargs)
