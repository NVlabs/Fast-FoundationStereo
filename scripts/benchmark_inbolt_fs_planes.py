"""Benchmark FastFoundationStereo models + RealSense hardware depth on the Inbolt dataset.

This benchmark mirrors the structure of ``benchmark_faro_rs.py`` but uses the
Inbolt dataset and the meter-based reporting pipeline already used by
``benchmark_inbolt.py``.

For fair pixel-wise comparison against the RealSense stereo pair and hardware
RealSense depth map, Zivid ground-truth depth is projected into RealSense image
space via ``DataSource.get_item_projected()``.

Usage:
  cd /home/adiroha/repos/Fast-FoundationStereo
  python scripts/benchmark_inbolt_fs.py [--out_dir reports/inbolt_ffs_benchmark]
"""

import argparse
import logging
import os
import sys
import time
import cv2
from pathlib import Path
from typing import Dict, Optional

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Utils as U
from benchmark_inbolt import DepthBinAccumulator, infer_depth_m, load_model, plot_depth_vs_distance
from scripts.data_manager_inbolt import DataSource, CAMERA_MATRIX_RS, DIST_COEFFS_RS
from metrics import (
    BenchmarkResults,
    FrameMetrics,
    compute_bin_mae,
    compute_metrics,
    aggregate,
    CLOSE_RANGE_THRESHOLD_M,
)
from report import ReportGenerator
from finetune_inbolt_planes import find_flat_regions


# ── custom report generator ──────────────────────────────────────────────────

class ReportGeneratorInbolt(ReportGenerator):
    """Custom report generator that shows 4 frames in depth comparison and error maps."""

    def __init__(self, results, stats, output_dir) -> None:
        super().__init__(results, stats, output_dir)
        self._selected_viz_indices = []

    def _get_selected_viz_indices(self, n_pick: int = 4):
        """Return cached random frame indices used consistently across report sections."""
        if self._selected_viz_indices:
            return self._selected_viz_indices

        n_total = len(self._r.viz_frames)
        if n_total == 0:
            self._selected_viz_indices = []
            return self._selected_viz_indices

        n = min(n_pick, n_total)
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
                im = ax.imshow(vf[name], cmap=cmap, vmin=0.1, vmax=2.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="m")
                title = self._r.method_labels.get(name, name)
                if c == 0:
                    title = f"Frame {frame_idx + 1} • {title}"
                ax.set_title(title, fontsize=9, wrap=True)
                ax.axis("off")

        fig.suptitle("Depth Map Comparison (4 random frames) — values in meters",
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
                im = ax.imshow(err, cmap=cmap, vmin=0.001, vmax=0.1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|error| (m)")
                mean_err = float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else 0.0
                label = self._r.method_labels.get(name, name)
                if c == 0:
                    ax.set_title(f"Frame {frame_idx + 1} • {label}\nMAE={mean_err:.4f} m", fontsize=9)
                else:
                    ax.set_title(f"{label}\nMAE={mean_err:.4f} m", fontsize=9)
                ax.axis("off")

        gt_label = self._r.method_labels.get(self._gt, self._gt)
        fig.suptitle(f"Absolute Error vs {gt_label} (4 random frames, m)", fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, "error_maps.png")


# ── constants ────────────────────────────────────────────────────────────────

DATA_DIR       = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260415T084601Z-3-001/Data Collection'
ORIGINAL_PATH  = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
# FINETUNED_PATH  = f'{code_dir}/../weights/20-30-48/model_finetuned_inbolt-20260415_epoch_030.pth'
# MODEL_PATH      = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
#FINETUNED_PATH  = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt-20260415_epoch_111.pth'
#DEFAULT_OUT     = f'{code_dir}/../reports/inbolt_ffs_benchmark-model37-111-set-20260414_142239'
FINETUNED_PATH  = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt_planes_25_epoch_012.pth'
DEFAULT_OUT     = f'{code_dir}/../reports/inbolt_ffs_benchmark-planes_25'
N_VIZ = 5

METHODS: Dict[str, Dict[str, str]] = {
    'original': {'label': 'FFS Original', 'color': '#2980b9'},
    'finetuned': {'label': 'FFS Fine-tuned (INBOLT)', 'color': '#e74c3c'},
    'depth_rs': {'label': 'RealSense Hardware Depth', 'color': '#f39c12'},
    'zivid_gt': {'label': 'Zivid GT (projected to RS)', 'color': '#27ae60'},
}
GT_NAME = 'zivid_gt'
RS_NAME = 'depth_rs'
RS_FPS = 30.0


def resolve_finetuned_model_path(preferred_path: str) -> Optional[str]:
    """Return an existing fine-tuned Inbolt checkpoint path, or None if not found."""
    preferred = Path(preferred_path)
    if preferred.exists():
        return str(preferred)

    weights_dir = Path(code_dir) / '..' / 'weights'
    candidate_names = [
        'model_finetuned_inbolt.pth',
        'model_finetuned_inbolt-20260415_epoch_030.pth',
    ]

    # 1) Try known candidate file names anywhere under weights/
    for name in candidate_names:
        found = sorted(weights_dir.glob(f'**/{name}'))
        if found:
            logging.warning(
                f'Preferred fine-tuned model not found at {preferred}. Using fallback {found[0]}'
            )
            return str(found[0])

    # 2) Fallback to any Inbolt fine-tuned checkpoint, prefer lexicographically latest
    generic = sorted(weights_dir.glob('**/model_finetuned_inbolt*.pth'))
    if generic:
        chosen = generic[-1]
        logging.warning(
            f'Preferred fine-tuned model not found at {preferred}. Using discovered checkpoint {chosen}'
        )
        return str(chosen)

    return None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out_dir', default=DEFAULT_OUT, help='Output directory for the report')
    parser.add_argument('--data_dir', default=DATA_DIR, help='Path to dataset root')
    parser.add_argument('--original', default=ORIGINAL_PATH, help='Path to original model weights')
    parser.add_argument('--finetuned', default=FINETUNED_PATH, help='Path to fine-tuned model weights')
    parser.add_argument('--n_viz', type=int, default=N_VIZ, help='Frames saved for visual comparison')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load stereo models ───────────────────────────────────────────────────
    models = {}
    finetuned_path = resolve_finetuned_model_path(args.finetuned)
    if finetuned_path is not None:
        models['finetuned'] = load_model(finetuned_path)
    else:
        logging.warning(
            f'Fine-tuned model not found (preferred: {args.finetuned}) and no fallback checkpoint found — skipping'
        )

    models['original'] = load_model(args.original)

    active_methods = [GT_NAME, RS_NAME] + list(models.keys())

    # ── dataset ──────────────────────────────────────────────────────────────
    source = DataSource(train_mode = False)
    n = source.init_directory(input_rectified=args.data_dir)
    logging.info(f'Found {n} samples in {args.data_dir}')
    if n == 0:
        logging.error('No samples found — check DATA_DIR path')
        return

    # ── accumulators ─────────────────────────────────────────────────────────
    all_metrics = []
    viz_frames = []
    valid_acc = {}
    dist_bin_mae = {m: [] for m in active_methods}
    close_range_valid = {m: [] for m in active_methods}
    timing_ms_raw = {m: [] for m in models}
    H = W = None

    depth_acc_keys = ['zivid_gt', RS_NAME] + list(models.keys())
    depth_accs = {k: DepthBinAccumulator() for k in depth_acc_keys}

    for idx in range(n):
        data = source.get_item_projected(idx)
        left = data['left']
        right = data['right']
        gt_mm = data['depth_zivid'].astype(np.float32)
        rs_mm = data['depth_rs'].astype(np.float32)

        if H is None:
            H, W = gt_mm.shape[:2]
            for m in active_methods:
                valid_acc[m] = np.zeros((H, W), np.float32)

        gt_m = gt_mm / 1000.0
        rs_m = rs_mm / 1000.0

        # valid only for flat regions
        valid = (gt_m > 0) 
        valid = find_flat_regions(gt_mm, valid)
        gt_m[valid == False] = 0.0

        frame_depths = {GT_NAME: gt_m, RS_NAME: rs_m}
        for mname, model in models.items():
            t0 = time.monotonic()
            frame_depths[mname] = infer_depth_m(model, left, right)
            # save raw data to p.g images 16 bit PNGs for later analysis if needed
            #cv2.imwrite(str(out_dir / f'{mname}_{idx:03d}.png'), (frame_depths[mname] * 1000.0).astype(np.uint16))
            timing_ms_raw[mname].append((time.monotonic() - t0) * 1000.0)

        gt_close_mask = (gt_m > 0) & (gt_m < CLOSE_RANGE_THRESHOLD_M)
        n_close = int(gt_close_mask.sum())

        # # create point clouds for visualization
        # if idx % 10 == 0:
        #     for mname in active_methods:
        #         pred = frame_depths[mname]

        #         XYZ = source.project_camera_to_3d(pred, CAMERA_MATRIX_RS, DIST_COEFFS_RS)  # (N, 3) array of 3D points in Zivid camera space
        #         mname_path = os.path.join(out_dir, f'{mname}_{idx:03d}.ply')
        #         source.save_to_ply(XYZ/1000, mname_path) # save in meters for visualization


        for mname in active_methods:
            pred = frame_depths[mname]

            valid_acc[mname] += (pred > 0).astype(np.float32)

            if mname == GT_NAME:
                fm = FrameMetrics(
                    GT_NAME,
                    0.0,
                    0.0,
                    0.0,
                    100.0,
                    float((pred > 0).mean()) * 100.0,
                    0.0,
                    mae_pen=0.0,
                    mre_pen=0.0,
                )
            elif mname == RS_NAME:
                fm = compute_metrics(pred, gt_m, elapsed_ms=0.0, method_name=RS_NAME)
            else:
                fm = compute_metrics(pred, gt_m, timing_ms_raw[mname][-1], mname)

            all_metrics.append(fm)
            dist_bin_mae[mname].append(compute_bin_mae(pred, gt_m))

            close_cov = (
                float((pred[gt_close_mask] > 0).mean()) * 100.0
                if n_close > 0 else 0.0
            )
            close_range_valid[mname].append(close_cov)

        depth_accs['zivid_gt'].update(gt_m, gt_m)
        depth_accs[RS_NAME].update(rs_m, gt_m)
        for mname in models:
            depth_accs[mname].update(frame_depths[mname], gt_m)

        if idx < args.n_viz:
            viz_frames.append({k: v.copy() for k, v in frame_depths.items()})

        if (idx + 1) % 200 == 0 or (idx + 1) == n:
            logging.info(f'  {idx + 1}/{n} frames processed')

    for m in active_methods:
        valid_acc[m] /= max(n, 1)

    mean_timing = {
        m: float(np.mean(ts)) if ts else 0.0
        for m, ts in timing_ms_raw.items()
    }
    mean_timing[GT_NAME] = 0.0
    mean_timing[RS_NAME] = 1000.0 / RS_FPS

    method_configs = {
        'original': {'model_path': args.original},
        RS_NAME: {'source': f'RealSense hardware depth (~{RS_FPS:.0f} FPS)'},
        GT_NAME: {'source': 'Projected Zivid depth map used as Inbolt ground truth'},
    }
    if 'finetuned' in models and finetuned_path is not None:
        method_configs['finetuned'] = {'model_path': finetuned_path}

    results = BenchmarkResults(
        method_names=active_methods,
        method_labels={m: METHODS[m]['label'] for m in active_methods},
        method_colors={m: METHODS[m]['color'] for m in active_methods},
        ground_truth_name=GT_NAME,
        n_frames=n,
        width=W,
        height=H,
        all_metrics=all_metrics,
        viz_frames=viz_frames,
        coverage_maps=valid_acc,
        dist_bin_mae=dist_bin_mae,
        close_range_valid=close_range_valid,
        source=f'INBOLT dataset ({args.data_dir})',
        method_configs=method_configs,
    )

    stats = aggregate(results, mean_timing)
    if RS_NAME in stats:
        stats[RS_NAME].fps_mean = RS_FPS

    reporter = ReportGeneratorInbolt(results, stats, out_dir)
    reporter.generate()

    plot_colors = {
        'zivid_gt': METHODS[GT_NAME]['color'],
        RS_NAME: METHODS[RS_NAME]['color'],
        **{m: METHODS[m]['color'] for m in models if m in METHODS},
    }
    plot_labels = {
        'zivid_gt': 'Zivid GT (spatial spread)',
        RS_NAME: METHODS[RS_NAME]['label'],
        'original': METHODS['original']['label'],
        'finetuned': METHODS['finetuned']['label'],
    }
    labeled_accs = {
        plot_labels.get(k, k): v
        for k, v in depth_accs.items()
        if depth_accs[k].count.sum() > 0
    }
    labeled_colors = {
        plot_labels.get(k, k): plot_colors.get(k)
        for k in depth_accs
        if depth_accs[k].count.sum() > 0
    }

    plot_depth_vs_distance(
        accumulators=labeled_accs,
        colors=labeled_colors,
        out_path=out_dir / 'depth_vs_distance.png',
    )
    logging.info(f'All outputs written to {out_dir}')


if __name__ == '__main__':
    main()
