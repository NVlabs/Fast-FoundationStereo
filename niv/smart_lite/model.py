"""
Smart Lite Stereo — lightweight stereo-matching with sensor-prior fusion.
Three sizes (S / M / L) share the same 4-stage architecture; only channel
widths and GRU iteration counts differ.

Stages:
  1. Depthwise-separable Siamese feature backbone  (→ 1/4 res)
  2. 8-group GWC cost volume + tiny 3-D head → coarse disparity (soft-argmax)
  3. DS-ConvGRU iterative refinement with 1-D bilinear correlation lookup
  4. Learned convex upsampling (3×3 mask head)

Quick start:
    from smart_lite.model import build_model
    model = build_model("M")          # or "S" / "L"
    model = build_model("L", max_disp=192)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Size presets
# ---------------------------------------------------------------------------

MODEL_CONFIGS = {
    # ── SmartLite-S ─────────────────────────────────────────────────────────
    # Baseline distillation target.  Fastest inference; fits comfortably on
    # Jetson Orin NX 8 GB.
    "S": dict(
        feature_ch=32,
        backbone_mid1=16, backbone_mid2=24, backbone_extra_blocks=0,
        hidden_dim=16, context_dim=24, context_mid=64, motion_ch=64,
        num_groups=8, corr_radius=4, num_iters=4,
    ),
    # ── SmartLite-M ─────────────────────────────────────────────────────────
    # Better accuracy, still real-time on a desktop GPU.
    "M": dict(
        feature_ch=48,
        backbone_mid1=24, backbone_mid2=36, backbone_extra_blocks=0,
        hidden_dim=32, context_dim=48, context_mid=96, motion_ch=96,
        num_groups=8, corr_radius=4, num_iters=6,
    ),
    # ── SmartLite-L ─────────────────────────────────────────────────────────
    # Highest accuracy among the three; one extra backbone refinement block.
    # Still far lighter than FoundationStereo.
    "L": dict(
        feature_ch=64,
        backbone_mid1=32, backbone_mid2=48, backbone_extra_blocks=1,
        hidden_dim=64, context_dim=64, context_mid=128, motion_ch=128,
        num_groups=8, corr_radius=4, num_iters=8,
    ),
}


def build_model(size: str = "S", max_disp: int = 192) -> "SmartLiteStereo":
    """Construct a SmartLiteStereo from a named size preset (S / M / L)."""
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model size '{size}'. Choose from: {list(MODEL_CONFIGS)}")
    cfg = MODEL_CONFIGS[size]
    return SmartLiteStereo(max_disp=max_disp, **cfg)


# ---------------------------------------------------------------------------
# Building blocks
# ---------------------------------------------------------------------------

class DepthwiseSeparableConv2d(nn.Module):
    """Depthwise-separable convolution: depthwise 3×3 + pointwise 1×1."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False):
        super().__init__()
        self.depthwise = nn.Conv2d(in_ch, in_ch, kernel_size, stride, padding,
                                   groups=in_ch, bias=False)
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=bias)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class DSConvBnRelu(nn.Module):
    """Depthwise-separable conv + BatchNorm + ReLU."""

    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = DepthwiseSeparableConv2d(in_ch, out_ch, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


# ---------------------------------------------------------------------------
# Stage 1 — Feature Backbone (Siamese, depthwise-separable)
# ---------------------------------------------------------------------------

class FeatureBackbone(nn.Module):
    """
    Vertical-stack Siamese backbone.  Accepts *stacked* left+right images
    ``[B, 3, 2*H, W]`` and produces per-image features at 1/4 resolution.

    Architecture:
        stem   (3 → mid1_ch, stride 2)  → 1/2 res
        stage1 (mid1_ch → mid2_ch, stride 2) → 1/4 res
        stage2 (mid2_ch → feature_ch, stride 1) × (2 + extra_blocks) → 1/4 res
    """

    def __init__(
        self,
        out_ch: int = 32,
        mid1_ch: int = 16,
        mid2_ch: int = 24,
        extra_blocks: int = 0,
    ):
        super().__init__()
        self.stem   = DSConvBnRelu(3,       mid1_ch, stride=2)
        self.stage1 = DSConvBnRelu(mid1_ch, mid2_ch, stride=2)
        blocks = [
            DSConvBnRelu(mid2_ch, out_ch, stride=1),
            DSConvBnRelu(out_ch,  out_ch, stride=1),
        ]
        for _ in range(extra_blocks):
            blocks.append(DSConvBnRelu(out_ch, out_ch, stride=1))
        self.stage2 = nn.Sequential(*blocks)

    def forward(self, left_rgb, right_rgb):
        stacked = torch.cat([left_rgb, right_rgb], dim=2)   # [B, 3, 2H, W]
        x = self.stage2(self.stage1(self.stem(stacked)))    # [B, out_ch, H/2, W/4]
        feat_left, feat_right = x.chunk(2, dim=2)
        return feat_left.contiguous(), feat_right.contiguous()


# ---------------------------------------------------------------------------
# Stage 1b — Group-wise Correlation (GWC) Volume
# ---------------------------------------------------------------------------

def build_gwc_volume(feat_left, feat_right, max_disp, num_groups=8):
    """
    Build a group-wise correlation volume.

    Returns:
        volume: [B, num_groups, max_disp, H, W]
    """
    B, C, H, W = feat_left.shape
    assert C % num_groups == 0
    cpg = C // num_groups

    left  = F.normalize(feat_left,  dim=1).view(B, num_groups, cpg, H, W)
    right = F.normalize(feat_right, dim=1).view(B, num_groups, cpg, H, W)

    volume = feat_left.new_zeros(B, num_groups, max_disp, H, W)
    for d in range(max_disp):
        if d == 0:
            volume[:, :, d] = (left * right).sum(dim=2)
        else:
            volume[:, :, d, :, d:] = (left[:, :, :, :, d:] * right[:, :, :, :, :-d]).sum(dim=2)
    return volume


# ---------------------------------------------------------------------------
# Stage 2 — Coarse Disparity Regression ("Warm Start")
# ---------------------------------------------------------------------------

class CoarseDisparityHead(nn.Module):
    """
    Tiny 2-layer 3-D conv head → soft-argmax over disparity dimension.

    Input:  GWC volume [B, G, D, H, W]
    Output: coarse disparity [B, 1, H, W]
    """

    def __init__(self, in_ch=8, mid_ch=8, max_disp=48):
        super().__init__()
        self.max_disp = max_disp
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_ch),
            nn.ReLU(inplace=True),
        )
        self.conv2 = nn.Conv3d(mid_ch, 1, kernel_size=3, padding=1)

    def forward(self, volume):
        x = self.conv1(volume)
        logits = self.conv2(x).squeeze(1)               # [B, D, H, W]
        prob   = F.softmax(logits, dim=1)
        disp_vals = torch.arange(0, self.max_disp, dtype=prob.dtype,
                                 device=prob.device).view(1, -1, 1, 1)
        return (prob * disp_vals).sum(dim=1, keepdim=True)


# ---------------------------------------------------------------------------
# Stage 2b — Context Network
# ---------------------------------------------------------------------------

class ContextNet(nn.Module):
    """
    Processes *left* features → GRU initial state + context + attention.

    Architecture: 2×Conv3×3 + CAM + SAM → proj
    Outputs: net0 [B, hidden_dim, H, W],
             inp0 [B, context_dim, H, W],
             att0 [B, 1, H, W]
    """

    def __init__(self, in_ch=32, hidden_dim=16, context_dim=24, mid_ch=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(True),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_ch, mid_ch, 3, padding=1), nn.BatchNorm2d(mid_ch), nn.ReLU(True),
        )
        self.proj = nn.Conv2d(mid_ch, hidden_dim + context_dim + 1, 1)

        self.cam = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(mid_ch, max(1, mid_ch // 4), 1), nn.ReLU(True),
            nn.Conv2d(max(1, mid_ch // 4), mid_ch, 1), nn.Sigmoid(),
        )
        self.sam = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3), nn.Sigmoid(),
        )
        self.hidden_dim  = hidden_dim
        self.context_dim = context_dim

    def forward(self, feat_left):
        x = self.conv1(feat_left)
        x = x * self.cam(x)
        x = self.conv2(x)
        sa = self.sam(torch.cat([x.mean(1, keepdim=True), x.amax(1, keepdim=True)], dim=1))
        x = x * sa # (1,128,H/4,W/4)
        out  = self.proj(x)
        net0 = torch.tanh(out[:, :self.hidden_dim])
        inp0 = torch.relu(out[:, self.hidden_dim:self.hidden_dim + self.context_dim])
        att0 = torch.sigmoid(out[:, -1:])
        return net0, inp0, att0


# ---------------------------------------------------------------------------
# Stage 3 — 1-D Bilinear Correlation Lookup
# ---------------------------------------------------------------------------

class CorrLookup1D(nn.Module):
    """
    Sample 2*radius+1 correlation values from the GWC volume around
    the current disparity estimate via bilinear interpolation.

    Returns: corr_features [B, G*(2*radius+1), H, W]
    """

    def __init__(self, radius=4, num_groups=8):
        super().__init__()
        self.radius     = radius
        self.num_groups = num_groups
        offsets = torch.arange(-radius, radius + 1, dtype=torch.float32)
        self.register_buffer("offsets", offsets)

    def forward(self, volume, disp):
        B, G, D, H, W = volume.shape
        r = self.radius

        sample_d      = disp + self.offsets.view(1, -1, 1, 1)
        sample_d_norm = 2.0 * sample_d / max(D - 1, 1) - 1.0

        vol_flat  = volume.view(B * G, 1, D, H * W)
        S         = H * W
        n_samples = 2 * r + 1

        spatial_norm = torch.linspace(-1, 1, S, device=volume.device)
        spatial_norm = spatial_norm.view(1, 1, 1, S).expand(B, n_samples, 1, S)
        d_norm = sample_d_norm.view(B, n_samples, H, W).reshape(B, n_samples, 1, S)

        grid = torch.stack(
            [spatial_norm.expand(B, n_samples, 1, S).reshape(B, n_samples, S),
             d_norm.reshape(B, n_samples, S)],
            dim=-1,
        )
        grid = grid.unsqueeze(1).expand(B, G, n_samples, S, 2).reshape(B * G, n_samples, S, 2)

        sampled = F.grid_sample(vol_flat, grid, mode='bilinear',
                                padding_mode='zeros', align_corners=True)
        return sampled.view(B, G, n_samples, H, W).view(B, G * n_samples, H, W)


# ---------------------------------------------------------------------------
# Stage 3 — Motion Encoder + DS-ConvGRU
# ---------------------------------------------------------------------------

class MotionEncoder(nn.Module):
    """
    Encodes correlation features + current disparity + sensor prior
    into a motion feature for the GRU.

    motion_ch controls the internal width of all three paths.
    """

    def __init__(self, corr_ch, prior_ch=2, hidden_dim=16, motion_ch=64):
        super().__init__()
        disp_ch  = max(4, motion_ch // 4)
        prior_w  = max(4, motion_ch // 4)
        fuse_in  = motion_ch + disp_ch + prior_w

        self.corr_net = nn.Sequential(
            nn.Conv2d(corr_ch, motion_ch, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(motion_ch, motion_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.disp_net = nn.Sequential(
            nn.Conv2d(1, disp_ch, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(disp_ch, disp_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.prior_net = nn.Sequential(
            nn.Conv2d(prior_ch, prior_w, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.fuse = nn.Conv2d(fuse_in, hidden_dim - 1, 1)

    def forward(self, disp, corr, prior):
        c = self.corr_net(corr)
        d = self.disp_net(disp)
        p = self.prior_net(prior)
        fused = F.relu(self.fuse(torch.cat([c, d, p], dim=1)))
        return torch.cat([fused, disp], dim=1)


class DepthwiseSeparableConvGRU(nn.Module):
    """ConvGRU cell using depthwise-separable convolutions."""

    def __init__(self, hidden_dim, input_dim, kernel_size=3):
        super().__init__()
        total = hidden_dim + input_dim
        pad   = kernel_size // 2
        self.convz = DepthwiseSeparableConv2d(total, hidden_dim, kernel_size, padding=pad)
        self.convr = DepthwiseSeparableConv2d(total, hidden_dim, kernel_size, padding=pad)
        self.convq = DepthwiseSeparableConv2d(total, hidden_dim, kernel_size, padding=pad)

    def forward(self, h, x):
        hx = torch.cat([h, x], dim=1)
        z  = torch.sigmoid(self.convz(hx))
        r  = torch.sigmoid(self.convr(hx))
        q  = torch.tanh(self.convq(torch.cat([r * h, x], dim=1)))
        return (1 - z) * h + z * q


class DispHead(nn.Module):
    """Predicts a residual disparity delta from GRU hidden state."""

    def __init__(self, hidden_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, 1, 3, padding=1),
        )

    def forward(self, h):
        return self.conv(h)


# ---------------------------------------------------------------------------
# Stage 4 — Learned Convex Upsampling (3×3 mask head)
# ---------------------------------------------------------------------------

class ConvexUpsample(nn.Module):
    """
    Predict 9 softmax weights per coarse pixel for content-adaptive 4× upsampling.
    """

    def __init__(self, hidden_dim, upsample_factor=4):
        super().__init__()
        self.factor    = upsample_factor
        self.mask_head = nn.Conv2d(hidden_dim, upsample_factor ** 2 * 9, 3, padding=1)

    def forward(self, h, disp_low):
        B, _, H, W = disp_low.shape
        f = self.factor

        mask = self.mask_head(h).view(B, 1, 9, f, f, H, W)
        mask = F.softmax(mask, dim=2)

        disp_unfold = F.unfold(disp_low * f, 3, padding=1).view(B, 1, 9, 1, 1, H, W)
        up = (mask * disp_unfold).sum(dim=2)
        return up.permute(0, 1, 4, 2, 5, 3).reshape(B, 1, H * f, W * f)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------

class SmartLiteStereo(nn.Module):
    """
    Smart Lite Stereo network.  Use ``build_model(size)`` for the standard
    S / M / L presets, or construct directly for custom configurations.

    Args:
        max_disp:              max disparity at full resolution (default 192)
        feature_ch:            backbone output channels
        backbone_mid1:         backbone stem output channels
        backbone_mid2:         backbone stage-1 output channels
        backbone_extra_blocks: extra refinement blocks in backbone stage-2
        hidden_dim:            GRU hidden-state channels
        context_dim:           context channels injected each GRU iteration
        context_mid:           internal width of ContextNet convolutions
        motion_ch:             internal width of MotionEncoder paths
        num_groups:            GWC correlation groups
        corr_radius:           1-D lookup radius (2r+1 samples per pixel)
        num_iters:             default GRU iterations
    """

    def __init__(
        self,
        max_disp: int = 192,
        feature_ch: int = 32,
        backbone_mid1: int = 16,
        backbone_mid2: int = 24,
        backbone_extra_blocks: int = 0,
        hidden_dim: int = 16,
        context_dim: int = 24,
        context_mid: int = 64,
        motion_ch: int = 64,
        num_groups: int = 8,
        corr_radius: int = 4,
        num_iters: int = 4,
    ):
        super().__init__()
        self.max_disp    = max_disp
        self.max_disp_q4 = max_disp // 4
        self.hidden_dim  = hidden_dim
        self.context_dim = context_dim
        self.num_groups  = num_groups
        self.num_iters   = num_iters

        self.backbone = FeatureBackbone(
            out_ch=feature_ch,
            mid1_ch=backbone_mid1,
            mid2_ch=backbone_mid2,
            extra_blocks=backbone_extra_blocks,
        )
        self.coarse_head = CoarseDisparityHead(
            in_ch=num_groups, mid_ch=num_groups, max_disp=self.max_disp_q4,
        )
        self.context_net = ContextNet(
            in_ch=feature_ch, hidden_dim=hidden_dim,
            context_dim=context_dim, mid_ch=context_mid,
        )

        corr_ch = num_groups * (2 * corr_radius + 1)
        self.motion_encoder = MotionEncoder(
            corr_ch=corr_ch, prior_ch=2,
            hidden_dim=hidden_dim, motion_ch=motion_ch,
        )
        self.gru = DepthwiseSeparableConvGRU(
            hidden_dim=hidden_dim, input_dim=hidden_dim + context_dim,
        )
        self.disp_head  = DispHead(hidden_dim)
        self.corr_lookup = CorrLookup1D(radius=corr_radius, num_groups=num_groups)
        self.upsample    = ConvexUpsample(hidden_dim=hidden_dim, upsample_factor=4)

    # ------------------------------------------------------------------

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------

    def forward(
        self,
        left_rgb: torch.Tensor,
        right_rgb: torch.Tensor,
        rs_disp: torch.Tensor,
        conf: torch.Tensor,
        num_iters: int | None = None,
        test_mode: bool = False,
    ):
        """
        Args:
            left_rgb:  [B, 3, H, W]
            right_rgb: [B, 3, H, W]
            rs_disp:   [B, 1, H, W]   RealSense raw disparity (0 = invalid)
            conf:      [B, 1, H, W]   confidence (0 = invalid)
            num_iters: override self.num_iters at inference
            test_mode: True → return only final full-res disparity
        Returns:
            test_mode=True  → disparity [B, 1, H, W]
            test_mode=False → (coarse [B,1,H/4,W/4], list of full-res preds)
        """
        iters = num_iters if num_iters is not None else self.num_iters
        B, _, H, W = left_rgb.shape
        Hq, Wq = H // 4, W // 4

        feat_left, feat_right = self.backbone(left_rgb, right_rgb)
        gwc_volume = build_gwc_volume(feat_left, feat_right,
                                      self.max_disp_q4, self.num_groups)

        rs_disp_q4 = F.interpolate(rs_disp, (Hq, Wq), mode='bilinear',
                                   align_corners=False) * 0.25
        conf_q4    = F.interpolate(conf,    (Hq, Wq), mode='bilinear',
                                   align_corners=False)
        prior = torch.cat([rs_disp_q4, conf_q4], dim=1)

        coarse_disp = self.coarse_head(gwc_volume)
        has_sensor  = (conf_q4 > 0).float()
        disp = has_sensor * rs_disp_q4 + (1.0 - has_sensor) * coarse_disp

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

            disp_up = self.upsample(net, disp)
            disp_preds.append(disp_up)

        if test_mode:
            return disp_up

        return coarse_disp, disp_preds
