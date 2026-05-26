"""Benchmark stereo models + RealSense hardware depth against FARO scanner ground truth.

All depth values are in millimetres (mm) throughout this script.

Metric descriptions
-------------------
MAE (mm)            Mean Absolute Error — average |predicted − GT| in mm over valid pixels.
                    Lower is better.  Sensitive to large outlier errors.

MAE* / MRE* (pen.)  Penalised variants of MAE / MRE.  Pixels where the model produces
                    no depth (holes) are penalised: MAE* counts the full GT depth as the
                    error, MRE* counts a 100% relative error.  Rewards methods with high
                    coverage — a sensor that refuses to predict on anything has 100% MRE*.

RMSE (mm)           Root Mean Square Error — sqrt(mean(|pred − GT|²)).
                    Penalises large individual errors more than MAE.  Lower is better.

MRE (%)             Mean Relative Error — mean(|pred − GT| / GT) × 100.
                    Scale-independent; allows fair comparison across depth ranges.
                    Lower is better.

δ1 (%)              Inlier accuracy — % of valid pixels where
                    max(pred/GT, GT/pred) < 1.25.  Higher is better.
                    A single threshold commonly used in stereo depth benchmarks.

Coverage (%)        % of image pixels where both prediction and GT are non-zero.
                    Higher is better.  Sensors with many holes score lower.

Close-range cov.    Coverage restricted to pixels where GT < 550 mm.  Important for
                    near-field robotics and manipulation tasks.  Higher is better.

Latency (ms)        Wall-clock inference time per frame (model only; data loading excluded).
                    Hardware sensors (RealSense) are treated as fixed 30 FPS devices.

FPS                 Frames per second = 1000 / latency_ms.

Usage:
  cd /home/adiroha/repos/Fast-FoundationStereo
  python scripts/benchmark_faro_rs.py [--out_dir reports/benchmark_faro_rs]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from core.utils.utils import InputPadder
import Utils as U
from scripts.data_manager_faro import DataSource
from metrics import (
    BenchmarkResults,
    FrameMetrics,
    compute_metrics,
    aggregate,
)
from report import ReportGenerator


# ── constants ─────────────────────────────────────────────────────────────────

FARO_DIR       = r'/mnt/algonas/Local/Data/Stereo/Faro/FARO_DATA_BASE'
ORIGINAL_PATH  = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
FINETUNED_PATH = f'{code_dir}/../weights/20-30-48/model_finetuned_faro_kitchen_epoch_006_epoch_013.pth'
DEFAULT_OUT    = f'{code_dir}/../reports/benchmark_faro_rs_bathroom'
DATASET_KEYWORDS = ['BATHROOM'] # ['OFFICE', 'KITCHEN']  # filter dataset paths containing any of these (case-insensitive)

BF     = 49470.45   # focal_px × baseline_mm  (calibrated from RealSense stereo pair)
ITERS  = 8          # GRU update iterations
N_VIZ  = 5         # frames saved for visual comparison in the report

# Depth threshold for the "close-range" coverage metric — in mm
CLOSE_RANGE_THRESHOLD_MM = 550.0

# Distance bins used for the per-bin MAE curve — all in mm
DIST_BINS_MM: List[Tuple[float, float]] = [
    (0.0,    500.0),
    (500.0,  1000.0),
    (1000.0, 2000.0),
    (2000.0, 3000.0)
]
BIN_LABELS_MM  = ["0–500 mm", "500–1000 mm", "1000–2000 mm", "2000–3000 mm"]
BIN_CENTERS_MM = [250.0, 750.0, 1500.0, 2500.0]

METHODS: Dict[str, Dict[str, str]] = {
    "original":  {"label": "FFS Original",            "color": "#2980b9"},
    "finetuned": {"label": "FFS Fine-tuned (FARO)",    "color": "#e74c3c"},
    "depth_rs":  {"label": "RealSense Hardware Depth", "color": "#f39c12"},
    "faro_gt":   {"label": "FARO GT",                  "color": "#27ae60"},
}
GT_NAME = "faro_gt"
RS_NAME = "depth_rs"   # pre-recorded RealSense active-stereo depth from the dataset


# ── mm-based metric helpers ───────────────────────────────────────────────────

def compute_bin_mae_mm(pred_mm: np.ndarray, gt_mm: np.ndarray) -> List[float]:
    """MAE (mm) per distance bin; returns NaN for bins with no valid GT pixels."""
    result = []
    for lo, hi in DIST_BINS_MM:
        mask = (gt_mm >= lo) & (gt_mm < hi) & (gt_mm > 0) & (pred_mm > 0)
        if mask.sum() == 0:
            result.append(float("nan"))
        else:
            result.append(float(np.abs(pred_mm[mask] - gt_mm[mask]).mean()))
    return result


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
def infer_depth_mm(model, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Run stereo inference on an IR pair; return depth map in mm (H×W float32).

    BF = focal_px × baseline_mm, so  depth_mm = BF / disparity_px.
    No unit conversion needed — result is already in mm.
    """
    left_t, right_t = _preprocess_ir(left, right)
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)

    with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
        disp = model.forward(left_t, right_t, iters=ITERS, test_mode=True)

    disp = padder.unpad(disp.float())
    disp_np = disp.cpu().numpy().reshape(left.shape[:2]).clip(0, None)

    depth_mm = np.zeros_like(disp_np)
    valid = disp_np > 0
    depth_mm[valid] = BF / disp_np[valid]   # disparity → mm  (BF already in focal·mm)
    return depth_mm


def load_model(path: str):
    logging.info(f"Loading model from {path}")
    model = torch.load(path, map_location='cpu', weights_only=False)
    model.cuda().eval()
    return model


# ── mm-aware report generator ─────────────────────────────────────────────────

class ReportGeneratorMM(ReportGenerator):
    """ReportGenerator subclass with all axis labels and colorbars in mm."""

    # Bin constants injected by the benchmark script
    _bin_labels  = BIN_LABELS_MM
    _bin_centers = BIN_CENTERS_MM

    def __init__(self, results, stats, output_dir) -> None:
        super().__init__(results, stats, output_dir)
        self._selected_viz_indices: List[int] = []

    def _get_selected_viz_indices(self, n_pick: int = 4) -> List[int]:
        """Return cached random frame indices used consistently across report sections."""
        if self._selected_viz_indices:
            return self._selected_viz_indices

        n_total = len(self._r.viz_frames)
        if n_total == 0:
            self._selected_viz_indices = []
            return self._selected_viz_indices

        n = min(n_pick, n_total)
        # fixed seed for reproducible reports
        rng = np.random.default_rng(42)
        self._selected_viz_indices = sorted(rng.choice(n_total, size=n, replace=False).tolist())
        return self._selected_viz_indices

    def _fig_depth_comparison(self) -> str:
        if not self._r.viz_frames:
            return self._empty_fig("depth_comparison.png", "No viz frames")

        sel = self._get_selected_viz_indices(n_pick=4)
        if not sel:
            return self._empty_fig("depth_comparison.png", "No viz frames")

        vf0 = self._r.viz_frames[sel[0]]
        method_names = [n for n in self._r.method_names if n in vf0]
        nrows = len(sel)
        ncols = len(method_names)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        axes = np.atleast_2d(axes)
        cmap = self._depth_cmap()

        for r, frame_idx in enumerate(sel):
            vf = self._r.viz_frames[frame_idx]
            for c, name in enumerate(method_names):
                ax = axes[r, c]
                if name not in vf:
                    ax.axis("off")
                    continue
                im = ax.imshow(vf[name], cmap=cmap, vmin=1.0, vmax=5000.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="mm")
                title = self._r.method_labels.get(name, name)
                if c == 0:
                    title = f"Frame {frame_idx + 1} • {title}"
                ax.set_title(title, fontsize=9, wrap=True)
                ax.axis("off")

        fig.suptitle("Depth Map Comparison (4 random frames) — values in mm",
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, "depth_comparison.png")

    def _fig_error_maps(self) -> str:
        if not self._r.viz_frames or not self._non_gt:
            return self._empty_fig("error_maps.png", "No comparison methods")

        sel = self._get_selected_viz_indices(n_pick=4)
        if not sel:
            return self._empty_fig("error_maps.png", "No viz frames")

        vf0 = self._r.viz_frames[sel[0]]
        names = ([self._gt] if self._gt in vf0 else []) + [n for n in self._non_gt if n in vf0]
        if not names:
            return self._empty_fig("error_maps.png", "Ground truth not available in viz frame")

        nrows = len(sel)
        ncols = len(names)
        cmap = plt.get_cmap("hot").copy()
        cmap.set_under("#222222")
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        axes = np.atleast_2d(axes)

        for r, frame_idx in enumerate(sel):
            vf = self._r.viz_frames[frame_idx]
            gt = vf.get(self._gt)
            if gt is None:
                for c in range(ncols):
                    axes[r, c].axis("off")
                continue

            for c, name in enumerate(names):
                ax = axes[r, c]
                if name not in vf:
                    ax.axis("off")
                    continue
                pred = vf[name]
                valid = (gt > 0) & (pred > 0)
                err = np.where(valid, np.abs(pred - gt), 0.0).astype(np.float32)
                im = ax.imshow(err, cmap=cmap, vmin=1.0, vmax=100.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|error| (mm)")
                mean_err = float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else 0.0
                label = self._r.method_labels.get(name, name)
                if c == 0:
                    ax.set_title(f"Frame {frame_idx + 1} • {label}\nMAE={mean_err:.0f} mm", fontsize=9)
                else:
                    ax.set_title(f"{label}\nMAE={mean_err:.0f} mm", fontsize=9)
                ax.axis("off")

        gt_label = self._r.method_labels.get(self._gt, self._gt)
        fig.suptitle(f"Absolute Error vs {gt_label} (4 random frames, mm)", fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, "error_maps.png")

    def _fig_distance_error_curve(self) -> str:
        if not self._non_gt:
            return self._empty_fig("distance_error_curve.png", "No comparison methods")
        fig, ax = plt.subplots(figsize=(8, 5))
        for name in self._non_gt:
            bin_data = self._r.dist_bin_mae.get(name, [])
            if not bin_data:
                continue
            arr = np.array(bin_data)
            mean_per_bin = np.array([
                np.nanmean(arr[:, i]) if np.any(~np.isnan(arr[:, i])) else 0.0
                for i in range(arr.shape[1])
            ])
            color = self._r.method_colors.get(name, "#888")
            label = self._r.method_labels.get(name, name)
            ax.plot(self._bin_centers, mean_per_bin, marker="o", color=color,
                    label=label, linewidth=2, markersize=7)
        ax.set_xticks(self._bin_centers)
        ax.set_xticklabels(self._bin_labels, fontsize=9)
        ax.set_xlabel("Distance range", fontsize=10)
        ax.set_ylabel("Mean Absolute Error (mm)", fontsize=10)
        ax.set_title("Depth Error vs Distance", fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "distance_error_curve.png")

    def _fig_error_histograms(self) -> str:
        if not self._non_gt or not self._r.viz_frames:
            return self._empty_fig("error_histograms.png", "No comparison data")
        names = [n for n in self._non_gt
                 if any(n in vf and self._gt in vf for vf in self._r.viz_frames)]
        if not names:
            return self._empty_fig("error_histograms.png", "No viz data for comparison")
        n = len(names)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
        axes = np.asarray(axes).flatten().tolist()
        for ax, name in zip(axes, names):
            errors = []
            for vf in self._r.viz_frames:
                if name not in vf or self._gt not in vf:
                    continue
                pred, gt = vf[name], vf[self._gt]
                valid = (gt > 0) & (pred > 0)
                if valid.any():
                    errors.extend(np.abs(pred[valid] - gt[valid]).tolist())
            if not errors:
                ax.text(0.5, 0.5, "No data", transform=ax.transAxes,
                        ha="center", va="center", color="gray")
                continue
            color = self._r.method_colors.get(name, "#888")
            label = self._r.method_labels.get(name, name)
            ax.hist(errors, bins=50, range=(0.0, 500.0), color=color,
                    alpha=0.8, edgecolor="none")
            mean_e = float(np.mean(errors))
            ax.axvline(mean_e, color="red", linestyle="--", linewidth=1.5,
                       label=f"mean={mean_e:.0f} mm")
            ax.set_xlabel("Absolute error (mm)", fontsize=9)
            ax.set_ylabel("Pixel count", fontsize=9)
            ax.set_title(label, fontsize=9)
            ax.legend(fontsize=8)
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Per-Pixel Error Distribution (vs GT, viz frames)", fontsize=11)
        fig.tight_layout()
        return self._save(fig, "error_histograms.png")

    def _fig_summary_table(self) -> str:
        if not self._stats:
            return self._empty_fig("summary_table.png", "No stats")
        cols = ["Method", "MRE* (%)", "MRE (%)", "MAE (mm)", "δ1 (%)",
                "Coverage (%)", "FPS", "GPU %", "GT?"]
        gt_rows, other_rows = [], []
        for name, s in self._stats.items():
            is_gt = (name == self._gt)
            row = [
                s.label,
                "—" if is_gt else f"{s.mre_pen_mean * 100:.1f}",
                "—" if is_gt else f"{s.mre_mean * 100:.1f}",
                "—" if is_gt else f"{s.mae_mean:.1f}",
                "—" if is_gt else f"{s.delta1_mean:.1f}",
                f"{s.coverage_mean:.1f}",
                f"{s.fps_mean:.1f}" if s.fps_mean < 999 else "≈30",
                f"{s.gpu_load_mean:.0f}" if s.gpu_load_mean > 0 else "—",
                "★ GT" if is_gt else "",
            ]
            (gt_rows if is_gt else other_rows).append((name, row))
        ordered = gt_rows + other_rows
        cell_text = [r for _, r in ordered]
        n = len(ordered)
        fig, ax = plt.subplots(figsize=(14, 1.0 + 0.55 * n))
        ax.axis("off")
        table = ax.table(cellText=cell_text, colLabels=cols,
                         cellLoc="center", loc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.6)
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#2c3e50")
                cell.set_text_props(color="white", fontweight="bold")
            elif row % 2 == 0:
                cell.set_facecolor("#f7f7f7")
            cell.set_edgecolor("#cccccc")
            if row > 0 and cell_text[row - 1][-1] == "★ GT":
                cell.set_facecolor("#d5f5d5")
        ax.set_title("Depth Quality Summary (errors in mm)", fontsize=12,
                     pad=10, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "summary_table.png")

    def _fig_close_range_analysis(self) -> str:
        names = list(self._r.method_names)
        if not names:
            return self._empty_fig("close_range_analysis.png", "No methods")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        labels    = [self._r.method_labels.get(n, n) for n in names]
        coverages = [self._stats[n].close_range_coverage if n in self._stats else 0.0
                     for n in names]
        colors    = [self._r.method_colors.get(n, "#888") for n in names]
        bars = ax1.bar(labels, coverages, color=colors, alpha=0.85, edgecolor="white")
        ax1.bar_label(bars, labels=[f"{v:.1f}%" for v in coverages], padding=3, fontsize=7)
        ax1.set_ylabel(f"Coverage at < {CLOSE_RANGE_THRESHOLD_MM:.0f} mm (%)", fontsize=10)
        ax1.set_title(f"Close-Range Coverage (< {CLOSE_RANGE_THRESHOLD_MM:.0f} mm)", fontsize=11)
        ax1.tick_params(axis="x", rotation=45, labelsize=7)
        ax1.set_xticklabels(labels, ha="right")
        ax1.set_ylim(0, 115)
        ax1.grid(axis="y", alpha=0.3)
        for name in names:
            vals = self._r.close_range_valid.get(name, [])
            if not vals:
                continue
            color = self._r.method_colors.get(name, "#888")
            label = self._r.method_labels.get(name, name)
            ax2.plot(range(1, len(vals) + 1), vals, color=color,
                     label=label, alpha=0.8, linewidth=1.5)
        ax2.set_xlabel("Frame", fontsize=10)
        ax2.set_ylabel(f"Coverage at < {CLOSE_RANGE_THRESHOLD_MM:.0f} mm (%)", fontsize=10)
        ax2.set_title("Close-Range Coverage per Frame", fontsize=11)
        ax2.legend(fontsize=6, loc="best")
        ax2.grid(alpha=0.3)
        ax2.set_ylim(-5, 115)
        fig.suptitle("Close-Range Depth Analysis", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "close_range_analysis.png")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out_dir',   default=DEFAULT_OUT,    help='Output directory for the report')
    parser.add_argument('--faro_dir',  default=FARO_DIR,       help='Path to FARO dataset root')
    parser.add_argument('--original',  default=ORIGINAL_PATH,  help='Path to original model weights')
    parser.add_argument('--finetuned', default=FINETUNED_PATH, help='Path to fine-tuned model weights')
    parser.add_argument('--n_viz', type=int, default=N_VIZ,    help='Frames saved for visual comparison')
    parser.add_argument('--keywords',  nargs='+', default=DATASET_KEYWORDS,  help='Keywords to filter dataset paths (case-insensitive)')
    parser.add_argument('--split', default='train', choices=['train', 'test', 'all'],
                        help='Dataset split to evaluate')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load stereo models ────────────────────────────────────────────────────
    models = {}
    if Path(args.finetuned).exists():
        models["finetuned"] = load_model(args.finetuned)
    else:
        logging.warning(f"Fine-tuned model not found at {args.finetuned} — skipping")

    models["original"] = load_model(args.original)

    # active_methods includes GT, RS hardware, and all NN models
    active_methods = [GT_NAME, RS_NAME] + list(models.keys())

    # ── dataset ───────────────────────────────────────────────────────────────
    source = DataSource()
    n = source.init_directory(
        input_rectified=args.faro_dir,
        test_keywords=args.keywords,
        split=args.split,
    )
    logging.info(f"Found {n} samples in {args.faro_dir}")
    if n == 0:
        logging.error("No samples found — check FARO_DIR path")
        return

    # ── accumulators ──────────────────────────────────────────────────────────
    all_metrics       = []
    viz_frames        = []
    valid_acc         = {}     # initialised on first frame
    dist_bin_mae      = {m: [] for m in active_methods}
    close_range_valid = {m: [] for m in active_methods}
    timing_ms_raw     = {m: [] for m in models}   # only NN models have inference latency
    H = W = None

    for idx in range(n):
        data  = source.get_item(idx)
        left  = data['left']
        right = data['right']
        gt_mm = data['depth_faro'].astype(np.float32)   # FARO GT — already in mm
        rs_mm = data['depth_rs'].astype(np.float32)     # RealSense hardware — already in mm

        if H is None:
            H, W = gt_mm.shape[:2]
            for m in active_methods:
                valid_acc[m] = np.zeros((H, W), np.float32)

        # ── run NN inference (returns mm) ─────────────────────────────────────
        frame_depths = {GT_NAME: gt_mm, RS_NAME: rs_mm}
        for mname, model in models.items():
            t0 = time.monotonic()
            frame_depths[mname] = infer_depth_mm(model, left, right)
            timing_ms_raw[mname].append((time.monotonic() - t0) * 1000.0)

        # ── per-frame metrics (all values in mm) ──────────────────────────────
        gt_close_mask = (gt_mm > 0) & (gt_mm < CLOSE_RANGE_THRESHOLD_MM)
        n_close = int(gt_close_mask.sum())

        for mname in active_methods:
            pred = frame_depths[mname]
            valid_acc[mname] += (pred > 0).astype(np.float32)

            if mname == GT_NAME:
                fm = FrameMetrics(
                    GT_NAME, 0.0, 0.0, 0.0, 100.0,
                    float((pred > 0).mean()) * 100.0, 0.0,
                    mae_pen=0.0, mre_pen=0.0,
                )
            elif mname == RS_NAME:
                # RealSense hardware depth — compare against FARO GT (both in mm)
                fm = compute_metrics(pred, gt_mm, elapsed_ms=0.0, method_name=RS_NAME)
            else:
                fm = compute_metrics(pred, gt_mm, timing_ms_raw[mname][-1], mname)

            all_metrics.append(fm)
            dist_bin_mae[mname].append(compute_bin_mae_mm(pred, gt_mm))

            close_cov = (
                float((pred[gt_close_mask] > 0).mean()) * 100.0
                if n_close > 0 else 0.0
            )
            close_range_valid[mname].append(close_cov)

        if idx < args.n_viz:
            viz_frames.append({k: v.copy() for k, v in frame_depths.items()})

        if (idx + 1) % 200 == 0 or (idx + 1) == n:
            logging.info(f"  {idx + 1}/{n} frames processed")

    # normalise coverage maps to [0, 1]
    for m in active_methods:
        valid_acc[m] /= max(n, 1)

    # ── aggregate timing ──────────────────────────────────────────────────────
    mean_timing: Dict[str, float] = {
        m: float(np.mean(ts)) if ts else 0.0
        for m, ts in timing_ms_raw.items()
    }
    # FARO GT has no latency; RealSense is hardware 30 FPS
    mean_timing[GT_NAME] = 0.0
    mean_timing[RS_NAME] = 1000.0 / 30.0   # ~33 ms per frame at native 30 FPS

    # ── build BenchmarkResults ────────────────────────────────────────────────
    method_configs = {
        "original": {"model_path": args.original},
    }
    if "finetuned" in models:
        method_configs["finetuned"] = {"model_path": args.finetuned}
    method_configs[RS_NAME]  = {"source": "RealSense D435 hardware depth (30 FPS)"}
    method_configs[GT_NAME]  = {"source": "FARO Focus 3D scanner — reference GT"}

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
        source=f"FARO dataset  •  {args.faro_dir}  •  split={args.split}  •  filter={args.keywords}",
        method_configs=method_configs,
    )

    # aggregate() computes per-method summaries; pass RS as hardware (30 fps)
    stats = aggregate(results, mean_timing)
    # Fix fps for hardware depth sensor (aggregate() uses 1000/t_ms)
    if RS_NAME in stats:
        stats[RS_NAME].fps_mean = 30.0

    # ── generate report ───────────────────────────────────────────────────────
    reporter = ReportGeneratorMM(results, stats, out_dir)
    reporter.generate()


if __name__ == '__main__':
    main()
