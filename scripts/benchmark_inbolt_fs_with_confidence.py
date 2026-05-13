"""Benchmark FastFoundationStereo models on the Inbolt dataset, including the
confidence-head variant produced by finetune_inbolt_with_confidence.py.

Methods compared:
  original   – pretrained FFS (no fine-tuning)
  finetuned  – FFS fine-tuned on INBOLT (standard loss)
  confidence – FFS fine-tuned on INBOLT with confidence head
  depth_rs   – RealSense hardware depth (baseline)
  zivid_gt   – Zivid projected ground truth

Confidence maps (one per frame) are stored alongside depth maps and rendered in
an extra report figure.

Usage:
  cd /home/adiroha/repos/Fast-FoundationStereo
  python scripts/benchmark_inbolt_fs_with_confidence.py [--out_dir reports/...]
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

import Utils as U
from core.utils.utils import InputPadder

# Import confidence model classes into __main__ scope so torch.load can
# find them when unpickling a model saved during training (where they were
# also in __main__).
from scripts.finetune_inbolt_with_confidence import FastFoundationStereoWithConfidence, ConfidenceHead  # noqa: F401

from benchmark_inbolt import DepthBinAccumulator, plot_depth_vs_distance
from benchmark_inbolt_fs import ReportGeneratorInbolt, resolve_finetuned_model_path
from scripts.data_manager_inbolt import DataSource, CAMERA_MATRIX_RS, DIST_COEFFS_RS
from metrics import (
    BenchmarkResults,
    FrameMetrics,
    compute_bin_mae,
    compute_metrics,
    aggregate,
    CLOSE_RANGE_THRESHOLD_M,
)


# ── constants ────────────────────────────────────────────────────────────────

DATA_DIR         = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260415T084601Z-3-001/Data Collection'
ORIGINAL_PATH    = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
FINETUNED_PATH   = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt-20260415_epoch_111.pth'
CONFIDENCE_PATH  = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt_with_confidence-20260507_epoch_026.pth'
DEFAULT_OUT      = f'{code_dir}/../reports/inbolt_ffs_confidence_benchmark'
N_VIZ  = 5
ITERS  = 8
BF     = 50.102706998586 * 385.509887695312   # focal_px * baseline_mm
RS_FPS = 30.0

CONF_VIZ_KEY = '_conf_map'   # suffix appended to 'confidence' in viz_frames

METHODS: Dict[str, Dict] = {
    'original':   {'label': 'FFS Original',                 'color': '#2980b9'},
    'finetuned':  {'label': 'FFS Fine-tuned (INBOLT)',      'color': '#e74c3c'},
    'confidence': {'label': 'FFS + Confidence Head',        'color': '#8e44ad'},
    'depth_rs':   {'label': 'RealSense Hardware Depth',     'color': '#f39c12'},
    'zivid_gt':   {'label': 'Zivid GT (projected to RS)',   'color': '#27ae60'},
}
GT_NAME = 'zivid_gt'
RS_NAME = 'depth_rs'


# ── inference helpers ─────────────────────────────────────────────────────────

def _preprocess_ir(left: np.ndarray, right: np.ndarray):
    """Convert IR uint8 pair to float RGB tensors on CUDA."""
    def _to_t(img):
        img = np.clip(img.astype(np.float32), 0, 255)
        img = np.stack([img, img, img], axis=-1)
        return torch.as_tensor(img).float()[None].permute(0, 3, 1, 2).cuda()
    return _to_t(left), _to_t(right)


@torch.no_grad()
def infer_depth_m(model, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Standard inference for models whose forward returns a disparity tensor."""
    left_t, right_t = _preprocess_ir(left, right)
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)
    with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
        disp = model.forward(left_t, right_t, iters=ITERS, test_mode=True)
    disp = padder.unpad(disp.float())
    disp_np = disp.cpu().numpy().reshape(left.shape[:2]).clip(0, None)
    depth_m = np.zeros_like(disp_np)
    valid = disp_np > 0
    depth_m[valid] = (BF / disp_np[valid]) / 1000.0
    return depth_m


@torch.no_grad()
def infer_depth_and_conf_m(
    model: FastFoundationStereoWithConfidence,
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Inference for the confidence model; returns (depth_m, conf [0–1])."""
    left_t, right_t = _preprocess_ir(left, right)
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)
    with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
        disp, conf = model.forward(left_t, right_t, iters=ITERS, test_mode=True)
    disp = padder.unpad(disp.float())
    conf = padder.unpad(conf.float())
    disp_np = disp.cpu().numpy().reshape(left.shape[:2]).clip(0, None)
    conf_np = conf.cpu().numpy().reshape(left.shape[:2]).clip(0.0, 1.0)
    depth_m = np.zeros_like(disp_np)
    valid = disp_np > 0
    depth_m[valid] = (BF / disp_np[valid]) / 1000.0
    return depth_m, conf_np


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(path: str):
    logging.info(f"Loading model from {path}")
    model = torch.load(path, map_location='cpu', weights_only=False)
    model.cuda().eval()
    return model


def resolve_confidence_model_path(preferred: str) -> Optional[str]:
    """Return an existing confidence-model checkpoint path, or None."""
    p = Path(preferred)
    if p.exists():
        return str(p)
    weights_dir = Path(code_dir) / '..' / 'weights'
    candidates = sorted(weights_dir.glob('**/model_finetuned_inbolt_with_confidence*.pth'))
    if candidates:
        chosen = candidates[-1]
        logging.warning(f"Preferred confidence model not found at {preferred}. Using {chosen}")
        return str(chosen)
    return None


# ── custom report generator ───────────────────────────────────────────────────

class ReportGeneratorWithConfidence(ReportGeneratorInbolt):
    """Extends the INBOLT report with a confidence map visualisation panel."""

    def generate(self) -> None:
        fig_paths = [
            self._fig_depth_comparison(),
            self._fig_error_maps(),
            self._fig_confidence_maps(),
            self._fig_coverage_heatmaps(),
            self._fig_distance_error_curve(),
            self._fig_error_histograms(),
            self._fig_summary_table(),
            self._fig_close_range_analysis(),
            self._fig_timing_bars(),
        ]
        self._write_json()
        self._write_html([p for p in fig_paths if p])
        print(f"\nReport written to: {self._out / 'index.html'}")

    def _fig_confidence_maps(self) -> str:
        """Render per-frame confidence maps for the confidence model."""
        conf_key = f'confidence{CONF_VIZ_KEY}'
        frames_with_conf = [vf for vf in self._r.viz_frames if conf_key in vf]
        if not frames_with_conf:
            return self._empty_fig("confidence_maps.png", "No confidence maps recorded")

        sel = self._get_selected_viz_indices(n_pick=4)
        sel = [i for i in sel if conf_key in self._r.viz_frames[i]]
        if not sel:
            return self._empty_fig("confidence_maps.png", "No confidence maps in selected frames")

        ncols = 3   # left image (depth), confidence, zivid_gt (for reference)
        nrows = len(sel)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        axes = np.atleast_2d(axes)

        depth_cmap = self._depth_cmap()
        conf_cmap  = plt.get_cmap('RdYlGn')   # red = low conf, green = high conf

        col_titles = [
            METHODS['confidence']['label'],
            'Confidence (0 = invalid, 1 = valid)',
            METHODS[GT_NAME]['label'],
        ]

        for row_idx, frame_idx in enumerate(sel):
            vf = self._r.viz_frames[frame_idx]
            depth_conf = vf.get('confidence')
            conf_map   = vf.get(conf_key)
            depth_gt   = vf.get(GT_NAME)

            # Column 0: confidence model depth
            ax = axes[row_idx, 0]
            if depth_conf is not None:
                im = ax.imshow(depth_conf, cmap=depth_cmap, vmin=0.1, vmax=2.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m')
            else:
                ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_titles[0], fontsize=8)
            ax.set_ylabel(f'Frame {frame_idx + 1}', fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 1: confidence map
            ax = axes[row_idx, 1]
            if conf_map is not None:
                im = ax.imshow(conf_map, cmap=conf_cmap, vmin=0.0, vmax=1.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                mean_conf = float(conf_map.mean())
                ax.set_title(f'{col_titles[1]}\nmean={mean_conf:.3f}', fontsize=8)
            else:
                ax.axis('off')
                if row_idx == 0:
                    ax.set_title(col_titles[1], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

            # Column 2: Zivid GT
            ax = axes[row_idx, 2]
            if depth_gt is not None:
                im = ax.imshow(depth_gt, cmap=depth_cmap, vmin=0.1, vmax=2.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m')
            else:
                ax.axis('off')
            if row_idx == 0:
                ax.set_title(col_titles[2], fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])

        fig.suptitle(
            'Confidence Head Output — Depth, Confidence Map, and Zivid GT',
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        return self._save(fig, 'confidence_maps.png')


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out_dir',    default=DEFAULT_OUT,       help='Output directory for the report')
    parser.add_argument('--data_dir',   default=DATA_DIR,          help='Path to dataset root')
    parser.add_argument('--original',   default=ORIGINAL_PATH,     help='Path to original model weights')
    parser.add_argument('--finetuned',  default=FINETUNED_PATH,    help='Path to standard fine-tuned model weights')
    parser.add_argument('--confidence', default=CONFIDENCE_PATH,   help='Path to confidence-head model weights')
    parser.add_argument('--n_viz',      type=int, default=N_VIZ,   help='Frames saved for visual comparison')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── load models ───────────────────────────────────────────────────────────
    regular_models: Dict[str, object] = {}
    conf_models:    Dict[str, object] = {}

    regular_models['original'] = load_model(args.original)

    ft_path = resolve_finetuned_model_path(args.finetuned)
    if ft_path:
        regular_models['finetuned'] = load_model(ft_path)
    else:
        logging.warning(f'Standard fine-tuned model not found at {args.finetuned} — skipping')
        ft_path = args.finetuned

    conf_path = resolve_confidence_model_path(args.confidence)
    if conf_path:
        conf_models['confidence'] = load_model(conf_path)
    else:
        logging.warning(f'Confidence model not found at {args.confidence} — skipping')
        conf_path = args.confidence

    all_nn_models   = {**regular_models, **conf_models}
    active_methods  = [GT_NAME, RS_NAME] + list(all_nn_models.keys())

    # ── dataset ───────────────────────────────────────────────────────────────
    source = DataSource()
    n = source.init_directory(input_rectified=args.data_dir)
    logging.info(f'Found {n} samples in {args.data_dir}')
    if n == 0:
        logging.error('No samples found — check DATA_DIR path')
        return

    # ── accumulators ──────────────────────────────────────────────────────────
    all_metrics:        list[FrameMetrics]         = []
    viz_frames:         list[dict]                 = []
    valid_acc:          Dict[str, np.ndarray]      = {}
    dist_bin_mae:       Dict[str, list]            = {m: [] for m in active_methods}
    close_range_valid:  Dict[str, list]            = {m: [] for m in active_methods}
    timing_ms_raw:      Dict[str, list]            = {m: [] for m in all_nn_models}
    H = W = None

    depth_acc_keys = [GT_NAME, RS_NAME] + list(all_nn_models.keys())
    depth_accs = {k: DepthBinAccumulator() for k in depth_acc_keys}

    for idx in range(n):
        data   = source.get_item_projected(idx)
        left   = data['left']
        right  = data['right']
        gt_mm  = data['depth_zivid'].astype(np.float32)
        rs_mm  = data['depth_rs'].astype(np.float32)

        if H is None:
            H, W = gt_mm.shape[:2]
            for m in active_methods:
                valid_acc[m] = np.zeros((H, W), np.float32)

        gt_m = gt_mm / 1000.0
        rs_m = rs_mm / 1000.0

        frame_depths: Dict[str, np.ndarray] = {GT_NAME: gt_m, RS_NAME: rs_m}
        frame_confs:  Dict[str, np.ndarray] = {}

        for mname, model in regular_models.items():
            t0 = time.monotonic()
            frame_depths[mname] = infer_depth_m(model, left, right)
            timing_ms_raw[mname].append((time.monotonic() - t0) * 1000.0)

        for mname, model in conf_models.items():
            t0 = time.monotonic()
            depth_m, conf_np = infer_depth_and_conf_m(model, left, right)
            timing_ms_raw[mname].append((time.monotonic() - t0) * 1000.0)
            frame_depths[mname] = depth_m
            frame_confs[mname]  = conf_np

        gt_close_mask = (gt_m > 0) & (gt_m < CLOSE_RANGE_THRESHOLD_M)
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

        depth_accs[GT_NAME].update(gt_m, gt_m)
        depth_accs[RS_NAME].update(rs_m, gt_m)
        for mname in all_nn_models:
            depth_accs[mname].update(frame_depths[mname], gt_m)

        if idx < args.n_viz:
            vf = {k: v.copy() for k, v in frame_depths.items()}
            # Store confidence maps under a separate key so they don't pollute depth metrics
            for mname, conf_np in frame_confs.items():
                vf[f'{mname}{CONF_VIZ_KEY}'] = conf_np.copy()
            viz_frames.append(vf)

        if (idx + 1) % 200 == 0 or (idx + 1) == n:
            logging.info(f'  {idx + 1}/{n} frames processed')

    for m in active_methods:
        valid_acc[m] /= max(n, 1)

    mean_timing = {m: float(np.mean(ts)) if ts else 0.0 for m, ts in timing_ms_raw.items()}
    mean_timing[GT_NAME] = 0.0
    mean_timing[RS_NAME] = 1000.0 / RS_FPS

    method_configs = {
        'original':  {'model_path': args.original},
        RS_NAME:     {'source': f'RealSense hardware depth (~{RS_FPS:.0f} FPS)'},
        GT_NAME:     {'source': 'Projected Zivid depth map used as INBOLT ground truth'},
    }
    if 'finetuned' in regular_models and ft_path:
        method_configs['finetuned'] = {'model_path': ft_path}
    if 'confidence' in conf_models and conf_path:
        method_configs['confidence'] = {'model_path': conf_path}

    results = BenchmarkResults(
        method_names   = active_methods,
        method_labels  = {m: METHODS[m]['label'] for m in active_methods},
        method_colors  = {m: METHODS[m]['color']  for m in active_methods},
        ground_truth_name = GT_NAME,
        n_frames       = n,
        width          = W,
        height         = H,
        all_metrics    = all_metrics,
        viz_frames     = viz_frames,
        coverage_maps  = valid_acc,
        dist_bin_mae   = dist_bin_mae,
        close_range_valid = close_range_valid,
        source         = f'INBOLT dataset ({args.data_dir})',
        method_configs = method_configs,
    )

    stats = aggregate(results, mean_timing)
    if RS_NAME in stats:
        stats[RS_NAME].fps_mean = RS_FPS

    reporter = ReportGeneratorWithConfidence(results, stats, out_dir)
    reporter.generate()

    # ── depth vs distance plot ─────────────────────────────────────────────────
    plot_colors = {
        GT_NAME:      METHODS[GT_NAME]['color'],
        RS_NAME:      METHODS[RS_NAME]['color'],
        **{m: METHODS[m]['color'] for m in all_nn_models if m in METHODS},
    }
    plot_labels = {
        GT_NAME:      'Zivid GT (spatial spread)',
        RS_NAME:      METHODS[RS_NAME]['label'],
        'original':   METHODS['original']['label'],
        'finetuned':  METHODS['finetuned']['label'],
        'confidence': METHODS['confidence']['label'],
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

    # ── console summary ───────────────────────────────────────────────────────
    logging.info('\n── Depth summary (test set) ─────────────────────────────────────────')
    for mname in active_methods:
        if mname == GT_NAME:
            continue
        s = stats.get(mname)
        if s is None:
            continue
        logging.info(
            f"  {s.label:<35}  MAE={s.mae_mean*1000:.1f} mm  "
            f"MRE*={s.mre_pen_mean*100:.2f}%  coverage={s.coverage_mean:.1f}%  "
            f"FPS={s.fps_mean:.1f}"
        )

    logging.info(f'\nAll outputs written to {out_dir}')


if __name__ == '__main__':
    main()
