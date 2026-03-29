"""Benchmark original vs fine-tuned FastFoundationStereo on the FARO dataset.

Loads both models, runs inference on all FARO samples, computes depth quality
metrics against FARO scanner ground truth, and produces an HTML report.

Usage:
  cd /home/adiroha/repos/Fast-FoundationStereo
  python scripts/benchmark_faro.py [--out_dir reports/faro_benchmark]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import numpy as np
import torch

from core.utils.utils import InputPadder
import Utils as U
from faro_data_manager import DataSource
from metrics import (
    BenchmarkResults,
    FrameMetrics,
    compute_metrics,
    compute_bin_mae,
    aggregate,
    CLOSE_RANGE_THRESHOLD_M,
)
from report import ReportGenerator


# ── constants ────────────────────────────────────────────────────────────────

FARO_DIR       = r'/mnt/algonas/Local/Data/Stereo/Faro/FARO_DATA_BASE'
#FARO_DIR       = r'data/faro'  # local path to FARO dataset --- IGNORE ---
ORIGINAL_PATH  = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
FINETUNED_PATH = f'{code_dir}/../weights/20-30-48/model_finetuned_faro_kitchen.pth'
DEFAULT_OUT    = f'{code_dir}/../reports/faro_benchmark_office'

BF     = 49470.45   # focal_px * baseline_mm  (calibrated from camera)
ITERS  = 8          # GRU iterations
N_VIZ  = 5         # number of frames saved for visual comparison in report

METHODS = {
    "original":  {"label": "Original model",    "color": "#2980b9"},
    "finetuned": {"label": "Fine-tuned on FARO", "color": "#e74c3c"},
    "faro_gt":   {"label": "FARO GT",            "color": "#27ae60"},
}
GT_NAME = "faro_gt"


# ── inference helpers ─────────────────────────────────────────────────────────

def _preprocess_ir(left: np.ndarray, right: np.ndarray):
    """Convert uint16 IR images to CUDA float tensors (3-channel pseudo-RGB)."""
    left  = np.clip(left.astype(np.float32),  0, 255)
    right = np.clip(right.astype(np.float32), 0, 255)
    left  = np.stack([left,  left,  left],  axis=-1)   # H×W×3
    right = np.stack([right, right, right], axis=-1)
    left_t  = torch.as_tensor(left).float()[None].permute(0, 3, 1, 2).cuda()
    right_t = torch.as_tensor(right).float()[None].permute(0, 3, 1, 2).cuda()
    return left_t, right_t


@torch.no_grad()
def infer_depth_m(model, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Run stereo inference on an IR pair; return depth map in metres (H×W float32)."""
    left_t, right_t = _preprocess_ir(left, right)
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)

    with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
        disp = model.forward(left_t, right_t, iters=ITERS, test_mode=True)

    disp = padder.unpad(disp.float())
    disp_np = disp.cpu().numpy().reshape(left.shape[:2]).clip(0, None)

    depth_m = np.zeros_like(disp_np)
    valid = disp_np > 0
    depth_m[valid] = (BF / disp_np[valid]) / 1000.0   # disparity → mm → m
    return depth_m


def load_model(path: str):
    logging.info(f"Loading model from {path}")
    model = torch.load(path, map_location='cpu', weights_only=False)
    model.cuda().eval()
    return model


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', default=DEFAULT_OUT, help='Output directory for the report')
    parser.add_argument('--faro_dir', default=FARO_DIR, help='Path to FARO dataset root')
    parser.add_argument('--original', default=ORIGINAL_PATH, help='Path to original model weights')
    parser.add_argument('--finetuned', default=FINETUNED_PATH, help='Path to fine-tuned model weights')
    parser.add_argument('--n_viz', type=int, default=N_VIZ, help='Frames saved for visual comparison')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load models ──────────────────────────────────────────────────────────
    models = {}
    if Path(args.finetuned).exists():
        models["finetuned"] = load_model(args.finetuned)
    else:
        logging.warning(f"Fine-tuned model not found at {args.finetuned} — skipping")

    models["original"] = load_model(args.original)

    active_methods = [GT_NAME] + list(models.keys())

    # ── dataset ───────────────────────────────────────────────────────────────
    source = DataSource()
    n = source.init_directory(input_rectified=args.faro_dir,test_keywords=['OFFICE'], split='train')
    logging.info(f"Found {n} samples in {args.faro_dir}")
    if n == 0:
        logging.error("No samples found — check FARO_DIR path")
        return

    # ── accumulators ──────────────────────────────────────────────────────────
    all_metrics       = []
    viz_frames        = []
    valid_acc         = {}     # will be init on first frame
    dist_bin_mae      = {m: [] for m in active_methods}
    close_range_valid = {m: [] for m in active_methods}
    timing_ms_raw     = {m: [] for m in models}
    H = W = None

    for idx in range(n):
        data  = source.get_item(idx)
        left  = data['left']
        right = data['right']
        gt_mm = data['depth_faro'].astype(np.float32)

        if H is None:
            H, W = gt_mm.shape[:2]
            for m in active_methods:
                valid_acc[m] = np.zeros((H, W), np.float32)

        gt_m = gt_mm / 1000.0   # mm → m

        # run inference for each model
        frame_depths = {GT_NAME: gt_m}
        for mname, model in models.items():
            t0 = time.monotonic()
            frame_depths[mname] = infer_depth_m(model, left, right)
            timing_ms_raw[mname].append((time.monotonic() - t0) * 1000.0)

        # per-frame metrics
        gt_close_mask = (gt_m > 0) & (gt_m < CLOSE_RANGE_THRESHOLD_M)
        n_close = int(gt_close_mask.sum())

        for mname in active_methods:
            pred = frame_depths[mname]
            valid_acc[mname] += (pred > 0).astype(np.float32)

            if mname == GT_NAME:
                fm = FrameMetrics(GT_NAME, 0.0, 0.0, 0.0, 100.0,
                                  float((pred > 0).mean()) * 100.0, 0.0,
                                  mae_pen=0.0, mre_pen=0.0)
            else:
                fm = compute_metrics(pred, gt_m, timing_ms_raw[mname][-1], mname)
            all_metrics.append(fm)

            dist_bin_mae[mname].append(compute_bin_mae(pred, gt_m))

            close_cov = float((pred[gt_close_mask] > 0).mean()) * 100.0 if n_close > 0 else 0.0
            close_range_valid[mname].append(close_cov)

        if idx < args.n_viz:
            viz_frames.append({k: v.copy() for k, v in frame_depths.items()})

        if (idx + 1) % 200 == 0 or (idx + 1) == n:
            logging.info(f"  {idx + 1}/{n} frames processed")

    # normalise coverage maps to [0, 1]
    for m in active_methods:
        valid_acc[m] /= max(n, 1)

    # ── aggregate timing ──────────────────────────────────────────────────────
    mean_timing = {m: float(np.mean(ts)) if ts else 0.0 for m, ts in timing_ms_raw.items()}
    mean_timing[GT_NAME] = 0.0

    # ── build BenchmarkResults ────────────────────────────────────────────────
    results = BenchmarkResults(
        method_names=active_methods,
        method_labels={m: METHODS[m]["label"] for m in active_methods},
        method_colors={m: METHODS[m]["color"] for m in active_methods},
        ground_truth_name=GT_NAME,
        n_frames=n,
        width=W,
        height=H,
        all_metrics=all_metrics,
        viz_frames=viz_frames,
        coverage_maps=valid_acc,
        dist_bin_mae=dist_bin_mae,
        close_range_valid=close_range_valid,
        source=f"FARO dataset ({args.faro_dir})",
        method_configs={
            "original":  {"model_path": args.original},
            "finetuned": {"model_path": args.finetuned},
        },
    )

    stats = aggregate(results, mean_timing)

    # ── generate report ───────────────────────────────────────────────────────
    reporter = ReportGenerator(results, stats, out_dir)
    reporter.generate()


if __name__ == '__main__':
    main()
