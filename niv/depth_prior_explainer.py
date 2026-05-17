#!/usr/bin/env python3
"""
D1-EdgeNeXt-FFS: RS depth prior injection points
=================================================

Shows exactly where rs_disp and conf enter the network during inference.
Two injection points:
  1. Sensor-gated disparity initialisation (before first GRU iteration)
  2. GRU motion encoder — prior fed at every iteration

Run:
    python3 docs/depth_prior_explainer.py --ckpt weights/stage3_best.pt

Bundled sample images are used automatically.  Pass --output-dir to save
the predicted depth map.
"""

import argparse
import os
import sys
import cv2, matlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt


import numpy as np
import torch
import torch.nn.functional as F


_HERE     = os.path.dirname(os.path.abspath(__file__))
_FFS_ROOT = os.path.abspath(os.path.join(_HERE, '..'))
if _FFS_ROOT not in sys.path:
    sys.path.insert(0, _FFS_ROOT)

from niv.smart_lite.model import build_gwc_volume
from niv.smart_lite.ablation.d1_models import build_d1_model    

MODEL_H, MODEL_W = 384, 512


# =============================================================================
# Input helpers
# =============================================================================

def depth_mm_to_prior(depth_mm, focal_px, baseline_m):
    """(H,W) uint16 mm → rs_disp (1,1,384,512) + conf (1,1,384,512) float32."""
    #src_h, src_w = depth_mm.shape
    # y0 = (src_h - MODEL_H) // 2
    # x0 = (src_w - MODEL_W) // 2
    # crop  = depth_mm[y0:y0+MODEL_H, x0:x0+MODEL_W].astype(np.float32)
    z_m   = depth_mm / 1000.0
    valid = z_m > 0.0
    rs_disp = np.where(valid, focal_px * baseline_m / np.maximum(z_m, 1e-6), 0.0).astype(np.float32)
    conf    = valid.astype(np.float32)
    return rs_disp[np.newaxis, np.newaxis], conf[np.newaxis, np.newaxis]


def _load_image(path):
    """Load image as (H,W,3) uint8 BGR.  Accepts PNG/JPG or FARO .mat (key Il/Ir)."""
    if path.endswith(".mat"):
        import scipy.io as sio
        mat = sio.loadmat(path)
        for key in ("Il", "Ir", "img", "image"):
            if key in mat:
                mono = mat[key].astype(np.uint8)
                return np.stack([mono, mono, mono], axis=-1)
        raise SystemExit(f"Unknown keys in .mat: {list(mat.keys())}")
    import cv2
    img = cv2.imread(path)
    if img is None:
        raise SystemExit(f"Cannot read image: {path}")
    if img.ndim == 2:
        import cv2 as _cv2
        img = _cv2.cvtColor(img, _cv2.COLOR_GRAY2BGR)
    return img


def _load_depth_mm(path):
    """Load depth as (H,W) uint16 mm.  Accepts PNG uint16 or FARO .mat (key Z_im)."""
    if path.endswith(".mat"):
        import scipy.io as sio
        mat = sio.loadmat(path)
        for key in ("Z_im", "depth_mm", "depth"):
            if key in mat:
                return mat[key].astype(np.uint16)
        raise SystemExit(f"Unknown keys in .mat: {list(mat.keys())}")
    import cv2
    d = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if d is None:
        raise SystemExit(f"Cannot read depth: {path}")
    return d.astype(np.uint16)


def to_tensor(bgr):
    #y0 = (stream_h - MODEL_H) // 2
    #x0 = (stream_w - MODEL_W) // 2
    crop = bgr #[y0:y0+MODEL_H, x0:x0+MODEL_W]
    rgb  = crop[..., ::-1].astype(np.float32) / 255.0
    return torch.from_numpy(np.ascontiguousarray(rgb.transpose(2, 0, 1))).unsqueeze(0).to(device)

# =============================================================================
# Annotated forward pass — two injection points
# =============================================================================

def annotated_forward(model, left_t, right_t, rs_disp_t, conf_t, num_iters=8):
    """
    Runs inference and prints where rs_disp / conf affect computation.

    INJECTION POINT 1 — disparity initialisation
        Before any GRU iteration, an initial disparity estimate is formed.
        Where RS sensor has data (conf > 0), the initial disparity is taken
        directly from the sensor.  Where there are holes, it falls back to
        a coarse cost-volume regression.

        has_sensor = conf_q4 > 0
        disp_init  = has_sensor * rs_q4  +  (1 - has_sensor) * coarse_disp

    INJECTION POINT 2 — GRU motion encoder (every iteration)
        At each of the num_iters GRU steps, the prior is concatenated into
        the motion encoder alongside the current disparity and correlation:

            motion = motion_encoder(current_disp, corr, prior)

        Inside MotionEncoder:
            p      = prior_net(prior)               # 2-ch → prior_w-ch
            fused  = relu(fuse(cat([corr, disp, p])))
            motion = cat([fused, disp], dim=1)

        The prior is therefore not a one-time warm-start — it acts as a
        persistent guidance channel at every refinement step.
    """
    with torch.no_grad():
        B, _, H, W = left_t.shape
        Hq, Wq = H // 4, W // 4

        # Feature extraction and cost volume (no prior here)
        feat_left, feat_right = model._extract(left_t, right_t)
        
        gwc_volume = build_gwc_volume(feat_left, feat_right,
                                      model.max_disp_q4, model.num_groups)

        # Downsample prior to GRU resolution (1/4), scale disparity accordingly
        rs_q4   = F.interpolate(rs_disp_t, (Hq, Wq), mode='bilinear', align_corners=False) * 0.25
        conf_q4 = F.interpolate(conf_t,    (Hq, Wq), mode='bilinear', align_corners=False)
        prior   = torch.cat([rs_q4, conf_q4], dim=1)   # (B, 2, Hq, Wq)

        # ── INJECTION POINT 1: sensor-gated disparity initialisation ──────────
        coarse_disp = model.coarse_head(gwc_volume)
        has_sensor  = (conf_q4 > 0).float()
        disp_init   = has_sensor * rs_q4 + (1.0 - has_sensor) * coarse_disp

        sensor_pct = has_sensor.mean().item() * 100
        print(f"\n  [INJECTION 1] disparity initialisation")
        print(f"    RS-seeded pixels : {sensor_pct:.1f}%   (conf > 0 → init from sensor)")
        print(f"    cost-vol pixels  : {100-sensor_pct:.1f}%   (holes → init from coarse regression)")
        print(f"    init disp range  : [{disp_init.min():.2f}, {disp_init.max():.2f}] px  (at 1/4 res)")

        # Context features (no prior)
        net, inp, att = model.context_net(feat_left)

        # ── INJECTION POINT 2: prior injected into motion encoder every step ──
        print(f"\n  [INJECTION 2] GRU iterations — prior in motion encoder each step")
        print(f"    prior shape fed to motion_encoder: {tuple(prior.shape)}  (rs_q4 + conf_q4)")
        disp       = disp_init
        disp_preds = []
        for i in range(num_iters):
            disp   = disp.detach()
            corr   = model.corr_lookup(gwc_volume, disp)
            motion = model.motion_encoder(disp, corr, prior)   # ← prior here
            net    = model.gru(net, torch.cat([motion, inp * att], dim=1))
            delta  = model.disp_head(net)
            disp   = disp + delta
            pred   = model.upsample(net, disp)
            disp_preds.append(pred)
            print(f"    iter {i+1}/{num_iters}:  delta_mean={delta.abs().mean():.4f} px")

        final = disp_preds[-1]
        print(f"\n  final disparity: {tuple(final.shape)}  "
              f"range [{final.min():.2f}, {final.max():.2f}] px")
    return final


# =============================================================================
# Output saving
# =============================================================================

def _save_output(out_dir, disp, focal_px, baseline_m):


    os.makedirs(out_dir, exist_ok=True)
    d           = disp[0, 0].cpu().numpy()
    valid       = d > 0.5
    depth_m     = np.zeros_like(d)
    depth_m[valid] = focal_px * baseline_m / np.maximum(d[valid], 1e-4)

    depth_mm = (depth_m * 1000).clip(0, 65535).astype(np.uint16)
    cv2.imwrite(os.path.join(out_dir, "depth.png"), depth_mm)

    fig, ax = plt.subplots(figsize=(8, 5))
    vmax = np.percentile(depth_m[valid], 95) if valid.any() else 5.0
    im = ax.imshow(depth_m, cmap="plasma", vmin=0, vmax=vmax)
    ax.set_title("D1-EdgeNeXt-FFS depth (real RS prior)", fontsize=11)
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="depth (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "depth_colorized.png"), dpi=120)
    plt.close(fig)

    print(f"\n  Saved → {out_dir}/depth.png  (uint16 mm)")
    print(f"         → {out_dir}/depth_colorized.png")


# =============================================================================
# Main
# =============================================================================

def main():
    p = argparse.ArgumentParser(
        description="D1-EdgeNeXt-FFS: RS prior injection point explainer")
    p.add_argument("--pt",       default = f'{_FFS_ROOT}/weights/weights_niv/stage3_best.pt',                help="Path to stage3_best.pt")
    p.add_argument("--ffs-ckpt", default = f'{_FFS_ROOT}/weights/weights_niv/model_best_bp2_serialize.pth',  help="FFS backbone weights (default: weights/model_best_bp2_serialize.pth)")
    p.add_argument("--device",   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--left",     metavar="FILE", help="Left IR/RGB image (PNG or FARO .mat)")
    p.add_argument("--right",    metavar="FILE", help="Right IR/RGB image")
    p.add_argument("--depth",    metavar="FILE", help="Hardware depth: PNG uint16 mm or FARO .mat")
    p.add_argument("--focal",    type=float, default=420.0, metavar="PX")
    p.add_argument("--baseline", type=float, default=0.05,  metavar="M")
    p.add_argument("--output-dir", default = f'{_FFS_ROOT}/demo_data_out')
    args = p.parse_args()

    device          = torch.device(args.device)
    model_path      = args.ffs_ckpt #or os.path.join(_FFS_ROOT, "weights", "weights_niv","model_best_bp2_serialize.pth")
    
    # Load model
    model           = build_d1_model("edgenext", max_disp=192, num_iters=8, ffs_ckpt=model_path)
    ck              = torch.load(args.pt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    model.eval().to(device)
    total           = sum(p.numel() for p in model.parameters()) / 1e6
    frozen          = sum(p.numel() for p in model.parameters() if not p.requires_grad) / 1e6
    print(f"Model: {total:.1f}M params  ({frozen:.1f}M frozen backbone, {total-frozen:.1f}M trained)")

    # Input data
    if args.left and args.right:
        left_bgr  = _load_image(args.left)
        right_bgr = _load_image(args.right)
        focal_px, baseline_m = args.focal, args.baseline
        stream_h, stream_w = left_bgr.shape[:2]
        depth_mm = _load_depth_mm(args.depth) if args.depth else np.zeros((stream_h, stream_w), dtype=np.uint16)
        print(f"Input: {stream_w}×{stream_h}  focal={focal_px:.1f}px  baseline={baseline_m*1000:.1f}mm  "
              f"depth valid={100*(depth_mm>0).mean():.1f}%")
    else:
        _sample_dir  = os.path.join(_FFS_ROOT, "demo_data")
        _sample_l    = os.path.join(_sample_dir, "imageL_d16_000.png")
        _sample_r    = os.path.join(_sample_dir, "imageR_d16_000.png")
        _sample_d    = os.path.join(_sample_dir, "imageD_d16_000.png")
        _sample_cam  = os.path.join(_sample_dir, "camera.txt")
        if os.path.isfile(_sample_l) and os.path.isfile(_sample_r):
            left_bgr  = _load_image(_sample_l)
            right_bgr = _load_image(_sample_r)
            depth_mm  = _load_depth_mm(_sample_d) if os.path.isfile(_sample_d) else np.zeros(left_bgr.shape[:2], dtype=np.uint16)
            focal_px, baseline_m = 420.0, 0.05
            if os.path.isfile(_sample_cam):
                for line in open(_sample_cam):
                    if line.startswith("focal_px="):    focal_px    = float(line.split("=")[1])
                    elif line.startswith("baseline_m="): baseline_m = float(line.split("=")[1])
            stream_h, stream_w = left_bgr.shape[:2]
            if args.output_dir is None:
                args.output_dir = os.path.join(_FFS_ROOT, "demo_data_out")
            print(f"Using bundled sample images ({stream_w}×{stream_h})")
        else:
            raise SystemExit("Provide --left / --right, or ensure sample/ directory is present.")

    # Depth mm → prior tensors
    rs_disp_np, conf_np = depth_mm_to_prior(depth_mm, focal_px, baseline_m)
    rs_disp_t = torch.from_numpy(rs_disp_np).to(device)
    conf_t    = torch.from_numpy(conf_np).to(device)

    left_t  = to_tensor(left_bgr)
    right_t = to_tensor(right_bgr)

    print("\n" + "=" * 55)
    print("RS prior injection points")
    #print("=" * 55)
    final = annotated_forward(model, left_t, right_t, rs_disp_t, conf_t, num_iters=8)

    if args.output_dir:
        _save_output(args.output_dir, final, focal_px, baseline_m)


if __name__ == "__main__":
    #python3 docs/depth_prior_explainer.py --ckpt weights/stage3_best.pt
    main()
