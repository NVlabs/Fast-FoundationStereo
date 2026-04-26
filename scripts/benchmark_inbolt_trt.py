"""Benchmark original + fine-tuned FFS (PyTorch) and fine-tuned FFS-TRT (TensorRT FP16) on Inbolt.

Extends ``benchmark_inbolt_fs.py`` by adding a TensorRT FP16 model alongside the
PyTorch models and RealSense hardware depth, so you can compare accuracy and speed
between the full-precision PyTorch path and the compiled TRT engine.

Pass --rebuild_trt to export ONNX and compile TRT engines from the fine-tuned model
at the start of the run (requires ~10 min on first compile).  The engines are written
to --trt_dir and reused on subsequent runs without --rebuild_trt.

Usage:
  cd /home/adiroha/repos/Fast-FoundationStereo
  # First run: build engines then benchmark
  python scripts/benchmark_inbolt_trt.py --rebuild_trt [--trt_dir output/onnx_trt_ft]
  # Subsequent runs: reuse existing engines
  python scripts/benchmark_inbolt_trt.py [--trt_dir output/onnx_trt_ft] [--out_dir reports/inbolt_trt_benchmark]
"""

import argparse
import logging
import os
import sys
import time
import warnings
import cv2
from pathlib import Path
from typing import Dict, Optional

code_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(f'{code_dir}/../')
sys.path.append(code_dir)

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import Utils as U
from benchmark_inbolt import DepthBinAccumulator, plot_depth_vs_distance, BF, ITERS
from benchmark_inbolt import infer_depth_m, load_model
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
from core.foundation_stereo import TrtRunner, TrtFeatureRunner, TrtPostRunner, build_gwc_volume_triton
from make_trt_engine import build_engine


# ── constants ────────────────────────────────────────────────────────────────

DATA_DIR       = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260415T084601Z-3-001/Data Collection'
ORIGINAL_PATH  = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
FINETUNED_PATH = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt-20260415_epoch_111.pth'
DEFAULT_TRT_DIR = f'{code_dir}/../output/onnx_trt'
DEFAULT_OUT    = f'{code_dir}/../reports/inbolt_trt_benchmark'
N_VIZ = 5

RS_FPS = 30.0

METHODS: Dict[str, Dict[str, str]] = {
    'original':      {'label': 'FFS Original',                  'color': '#2980b9'},
    'finetuned':     {'label': 'FFS Fine-tuned (INBOLT)',        'color': '#e74c3c'},
    'finetuned_trt': {'label': 'FFS Fine-tuned TRT (FP16)',      'color': '#8e44ad'},
    'depth_rs':      {'label': 'RealSense Hardware Depth',       'color': '#f39c12'},
    'zivid_gt':      {'label': 'Zivid GT (projected to RS)',     'color': '#27ae60'},
}
GT_NAME = 'zivid_gt'
RS_NAME = 'depth_rs'


# ── TRT build ────────────────────────────────────────────────────────────────

def rebuild_trt_engines(model_path: str, trt_dir: str, height: int, width: int) -> None:
    """Export the model at *model_path* to ONNX, then compile both TRT engines.

    Writes to *trt_dir*:
        feature_runner.onnx / feature_runner.engine
        post_runner.onnx    / post_runner.engine
        onnx.yaml
    """
    assert height % 32 == 0 and width % 32 == 0, \
        f"height and width must be divisible by 32, got {height}x{width}"

    os.makedirs(trt_dir, exist_ok=True)
    logging.info(f'[rebuild] Loading model from {model_path}')
    model = torch.load(model_path, map_location='cpu', weights_only=False)
    model.cuda().eval()

    feature_runner = TrtFeatureRunner(model).cuda().eval()
    post_runner    = TrtPostRunner(model).cuda().eval()

    dummy_left  = torch.randn(1, 3, height, width, device='cuda').float() * 255
    dummy_right = torch.randn(1, 3, height, width, device='cuda').float() * 255

    # ── feature runner → ONNX ────────────────────────────────────────────────
    feature_onnx = os.path.join(trt_dir, 'feature_runner.onnx')
    logging.info(f'[rebuild] Exporting feature_runner → {feature_onnx}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.onnx.export(
            feature_runner,
            (dummy_left, dummy_right),
            feature_onnx,
            opset_version=17,
            input_names=['left', 'right'],
            output_names=['features_left_04', 'features_left_08', 'features_left_16',
                          'features_left_32', 'features_right_04', 'stem_2x'],
            do_constant_folding=True,
            dynamo=False,
        )

    # ── post runner → ONNX ───────────────────────────────────────────────────
    with torch.no_grad():
        feats = feature_runner(dummy_left, dummy_right)
        f04, f08, f16, f32, fr04, stem_2x = feats
        cv_group = getattr(model, 'cv_group', 8)
        gwc_volume = build_gwc_volume_triton(
            f04.half(), fr04.half(), model.args.max_disp // 4, cv_group
        )

    post_onnx = os.path.join(trt_dir, 'post_runner.onnx')
    logging.info(f'[rebuild] Exporting post_runner → {post_onnx}')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.onnx.export(
            post_runner,
            (f04.float(), f08.float(), f16.float(), f32.float(),
             fr04.float(), stem_2x.float(), gwc_volume.float()),
            post_onnx,
            opset_version=17,
            input_names=['features_left_04', 'features_left_08', 'features_left_16',
                         'features_left_32', 'features_right_04', 'stem_2x', 'gwc_volume'],
            output_names=['disp'],
            do_constant_folding=True,
            dynamo=False,
        )

    # ── save model config ─────────────────────────────────────────────────────
    yaml_path = os.path.join(trt_dir, 'onnx.yaml')
    with open(yaml_path, 'w') as f:
        yaml.safe_dump(OmegaConf.to_container(model.args), f)
    logging.info(f'[rebuild] Saved onnx.yaml → {yaml_path}')

    # ── compile TRT engines ───────────────────────────────────────────────────
    del model, feature_runner, post_runner  # free GPU memory before TRT build
    torch.cuda.empty_cache()

    for name in ('feature_runner', 'post_runner'):
        onnx_path   = os.path.join(trt_dir, f'{name}.onnx')
        engine_path = os.path.join(trt_dir, f'{name}.engine')
        logging.info(f'[rebuild] Compiling {name}.engine (this may take several minutes) …')
        build_engine(onnx_path, engine_path, fp16=True, workspace_gb=4)

    logging.info(f'[rebuild] TRT engines ready in {trt_dir}')


# ── TRT helpers ───────────────────────────────────────────────────────────────

def load_trt_model(trt_dir: str) -> Optional[TrtRunner]:
    """Load TRT engines from *trt_dir*; return None if engines are missing."""
    feature_engine = os.path.join(trt_dir, 'feature_runner.engine')
    post_engine    = os.path.join(trt_dir, 'post_runner.engine')
    yaml_path      = os.path.join(trt_dir, 'onnx.yaml')

    for p in (feature_engine, post_engine, yaml_path):
        if not os.path.exists(p):
            logging.warning(f'TRT file not found: {p} — skipping TRT model')
            return None

    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    args = OmegaConf.create(cfg)

    logging.info(f'Loading TRT engines from {trt_dir}')
    return TrtRunner(args, feature_engine, post_engine)


@torch.no_grad()
def infer_depth_m_trt(
    trt_model: TrtRunner,
    trt_h: int,
    trt_w: int,
    left: np.ndarray,
    right: np.ndarray,
) -> np.ndarray:
    """Run TRT stereo inference; return depth map in metres (H×W float32).

    Images are resized to the fixed TRT engine resolution, disparity is scaled
    back to original pixel units before the BF depth conversion.
    """
    orig_h, orig_w = left.shape[:2]
    fx = trt_w / orig_w

    left_r  = cv2.resize(left.astype(np.float32),  (trt_w, trt_h))
    right_r = cv2.resize(right.astype(np.float32), (trt_w, trt_h))

    # pseudo-RGB (same as _preprocess_ir in benchmark_inbolt.py)
    left_r  = np.stack([left_r,  left_r,  left_r],  axis=-1)
    right_r = np.stack([right_r, right_r, right_r], axis=-1)

    left_t  = torch.as_tensor(left_r).float()[None].permute(0, 3, 1, 2).cuda()
    right_t = torch.as_tensor(right_r).float()[None].permute(0, 3, 1, 2).cuda()

    disp = trt_model.forward(left_t, right_t)
    disp_np = disp.cpu().numpy().reshape(trt_h, trt_w).clip(0, None)

    # resize back to original resolution; divide by fx to restore pixel-unit disparity
    disp_orig = cv2.resize(disp_np, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR) / fx

    depth_m = np.zeros_like(disp_orig)
    valid = disp_orig > 0
    depth_m[valid] = (BF / disp_orig[valid]) / 1000.0
    return depth_m


# ── report generator ──────────────────────────────────────────────────────────

class ReportGeneratorInbolt(ReportGenerator):
    """4-frame depth comparison and error maps."""

    def __init__(self, results, stats, output_dir) -> None:
        super().__init__(results, stats, output_dir)
        self._selected_viz_indices = []

    def _get_selected_viz_indices(self, n_pick: int = 4):
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
            return self._empty_fig('depth_comparison.png', 'No viz frames')
        sel = self._get_selected_viz_indices(n_pick=4)
        if not sel:
            return self._empty_fig('depth_comparison.png', 'No viz frames')
        vf0 = self._r.viz_frames[sel[0]]
        method_names = [n for n in self._r.method_names if n in vf0]
        nrows, ncols = len(sel), len(method_names)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        axes = np.atleast_2d(axes)
        cmap = self._depth_cmap()
        for r, frame_idx in enumerate(sel):
            vf = self._r.viz_frames[frame_idx]
            for c, name in enumerate(method_names):
                ax = axes[r, c]
                if name not in vf:
                    ax.axis('off')
                    continue
                im = ax.imshow(vf[name], cmap=cmap, vmin=0.1, vmax=2.0)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='m')
                title = self._r.method_labels.get(name, name)
                if c == 0:
                    title = f'Frame {frame_idx + 1} • {title}'
                ax.set_title(title, fontsize=9, wrap=True)
                ax.axis('off')
        fig.suptitle('Depth Map Comparison (4 random frames) — values in metres',
                     fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, 'depth_comparison.png')

    def _fig_error_maps(self) -> str:
        if not self._r.viz_frames or not self._non_gt:
            return self._empty_fig('error_maps.png', 'No comparison methods')
        sel = self._get_selected_viz_indices(n_pick=4)
        if not sel:
            return self._empty_fig('error_maps.png', 'No viz frames')
        vf0 = self._r.viz_frames[sel[0]]
        names = ([self._gt] if self._gt in vf0 else []) + [n for n in self._non_gt if n in vf0]
        if not names:
            return self._empty_fig('error_maps.png', 'Ground truth not in viz frame')
        nrows, ncols = len(sel), len(names)
        cmap = plt.get_cmap('hot').copy()
        cmap.set_under('#222222')
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.8 * nrows))
        axes = np.atleast_2d(axes)
        for r, frame_idx in enumerate(sel):
            vf = self._r.viz_frames[frame_idx]
            gt = vf.get(self._gt)
            if gt is None:
                for c in range(ncols):
                    axes[r, c].axis('off')
                continue
            for c, name in enumerate(names):
                ax = axes[r, c]
                if name not in vf:
                    ax.axis('off')
                    continue
                pred  = vf[name]
                valid = (gt > 0) & (pred > 0)
                err   = np.where(valid, np.abs(pred - gt), 0.0).astype(np.float32)
                im    = ax.imshow(err, cmap=cmap, vmin=0.001, vmax=0.1)
                plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='|error| (m)')
                mean_err = float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else 0.0
                label = self._r.method_labels.get(name, name)
                title = f'Frame {frame_idx + 1} • {label}\nMAE={mean_err:.4f} m' if c == 0 \
                        else f'{label}\nMAE={mean_err:.4f} m'
                ax.set_title(title, fontsize=9)
                ax.axis('off')
        gt_label = self._r.method_labels.get(self._gt, self._gt)
        fig.suptitle(f'Absolute Error vs {gt_label} (4 random frames, m)', fontsize=11, y=1.01)
        fig.tight_layout()
        return self._save(fig, 'error_maps.png')


# ── misc helpers ──────────────────────────────────────────────────────────────

def resolve_finetuned_model_path(preferred_path: str) -> Optional[str]:
    preferred = Path(preferred_path)
    if preferred.exists():
        return str(preferred)
    weights_dir = Path(code_dir) / '..' / 'weights'
    for name in ('model_finetuned_inbolt.pth', 'model_finetuned_inbolt-20260415_epoch_030.pth'):
        found = sorted(weights_dir.glob(f'**/{name}'))
        if found:
            logging.warning(f'Preferred fine-tuned model not found. Using fallback {found[0]}')
            return str(found[0])
    generic = sorted(weights_dir.glob('**/model_finetuned_inbolt*.pth'))
    if generic:
        chosen = generic[-1]
        logging.warning(f'Preferred fine-tuned model not found. Using discovered checkpoint {chosen}')
        return str(chosen)
    return None


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument('--out_dir',   default=DEFAULT_OUT,    help='Output directory')
    parser.add_argument('--data_dir',  default=DATA_DIR,       help='Path to dataset root')
    parser.add_argument('--original',  default=ORIGINAL_PATH,  help='Original model weights')
    parser.add_argument('--finetuned', default=FINETUNED_PATH, help='Fine-tuned model weights')
    parser.add_argument('--trt_dir',   default=DEFAULT_TRT_DIR,
                        help='Directory with feature_runner.engine, post_runner.engine, onnx.yaml')
    parser.add_argument('--trt_height', type=int, default=448, help='TRT engine input height')
    parser.add_argument('--trt_width',  type=int, default=640, help='TRT engine input width')
    parser.add_argument('--rebuild_trt', action='store_true',
                        help='Re-export ONNX and recompile TRT engines from --finetuned before benchmarking')
    parser.add_argument('--n_viz', type=int, default=N_VIZ, help='Frames saved for visual comparison')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── optionally rebuild TRT engines ────────────────────────────────────────
    if args.rebuild_trt:
        build_src = resolve_finetuned_model_path(args.finetuned) or args.original
        logging.info(f'[rebuild_trt] Building TRT engines from {build_src} → {args.trt_dir}')
        rebuild_trt_engines(build_src, args.trt_dir, args.trt_height, args.trt_width)

    # ── load PyTorch models ───────────────────────────────────────────────────
    pt_models = {}
    finetuned_path = resolve_finetuned_model_path(args.finetuned)
    if finetuned_path is not None:
        pt_models['finetuned'] = load_model(finetuned_path)
    else:
        logging.warning(f'Fine-tuned model not found (preferred: {args.finetuned}) — skipping')
    pt_models['original'] = load_model(args.original)

    # ── load TRT model ────────────────────────────────────────────────────────
    trt_model = load_trt_model(args.trt_dir)
    trt_h, trt_w = args.trt_height, args.trt_width

    # ── assemble active methods ───────────────────────────────────────────────
    active_methods = [GT_NAME, RS_NAME] + list(pt_models.keys())
    if trt_model is not None:
        active_methods.append('finetuned_trt')

    logging.info(f'Active methods: {active_methods}')

    # ── dataset ───────────────────────────────────────────────────────────────
    source = DataSource()
    n = source.init_directory(input_rectified=args.data_dir)
    logging.info(f'Found {n} samples in {args.data_dir}')
    if n == 0:
        logging.error('No samples found — check DATA_DIR path')
        return

    # ── accumulators ──────────────────────────────────────────────────────────
    all_metrics       = []
    viz_frames        = []
    valid_acc         = {}
    dist_bin_mae      = {m: [] for m in active_methods}
    close_range_valid = {m: [] for m in active_methods}
    timing_ms_raw     = {m: [] for m in active_methods if m not in (GT_NAME, RS_NAME)}
    H = W = None

    depth_acc_keys = [GT_NAME, RS_NAME] + [m for m in active_methods if m not in (GT_NAME, RS_NAME)]
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

        frame_depths = {GT_NAME: gt_m, RS_NAME: rs_m}

        # PyTorch models
        for mname, model in pt_models.items():
            t0 = time.monotonic()
            frame_depths[mname] = infer_depth_m(model, left, right)
            cv2.imwrite(str(out_dir / f'{mname}_{idx:03d}.png'),
                        (frame_depths[mname] * 1000.0).astype(np.uint16))
            timing_ms_raw[mname].append((time.monotonic() - t0) * 1000.0)

        # TRT model
        if trt_model is not None:
            t0 = time.monotonic()
            frame_depths['finetuned_trt'] = infer_depth_m_trt(trt_model, trt_h, trt_w, left, right)
            cv2.imwrite(str(out_dir / f'finetuned_trt_{idx:03d}.png'),
                        (frame_depths['finetuned_trt'] * 1000.0).astype(np.uint16))
            timing_ms_raw['finetuned_trt'].append((time.monotonic() - t0) * 1000.0)

        gt_close_mask = (gt_m > 0) & (gt_m < CLOSE_RANGE_THRESHOLD_M)
        n_close = int(gt_close_mask.sum())

        for mname in active_methods:
            pred = frame_depths[mname]
            valid_acc[mname] += (pred > 0).astype(np.float32)

            if mname == GT_NAME:
                fm = FrameMetrics(GT_NAME, 0.0, 0.0, 0.0, 100.0,
                                  float((pred > 0).mean()) * 100.0, 0.0,
                                  mae_pen=0.0, mre_pen=0.0)
            elif mname == RS_NAME:
                fm = compute_metrics(pred, gt_m, elapsed_ms=0.0, method_name=RS_NAME)
            else:
                fm = compute_metrics(pred, gt_m, timing_ms_raw[mname][-1], mname)

            all_metrics.append(fm)
            dist_bin_mae[mname].append(compute_bin_mae(pred, gt_m))
            close_cov = (float((pred[gt_close_mask] > 0).mean()) * 100.0
                         if n_close > 0 else 0.0)
            close_range_valid[mname].append(close_cov)

        depth_accs[GT_NAME].update(gt_m, gt_m)
        depth_accs[RS_NAME].update(rs_m, gt_m)
        for mname in active_methods:
            if mname not in (GT_NAME, RS_NAME):
                depth_accs[mname].update(frame_depths[mname], gt_m)

        if idx < args.n_viz:
            viz_frames.append({k: v.copy() for k, v in frame_depths.items()})

        if (idx + 1) % 200 == 0 or (idx + 1) == n:
            logging.info(f'  {idx + 1}/{n} frames processed')

    for m in active_methods:
        valid_acc[m] /= max(n, 1)

    mean_timing = {m: float(np.mean(ts)) if ts else 0.0
                   for m, ts in timing_ms_raw.items()}
    mean_timing[GT_NAME] = 0.0
    mean_timing[RS_NAME] = 1000.0 / RS_FPS

    method_configs = {
        'original':  {'model_path': args.original},
        RS_NAME:     {'source': f'RealSense hardware depth (~{RS_FPS:.0f} FPS)'},
        GT_NAME:     {'source': 'Projected Zivid depth map used as Inbolt ground truth'},
    }
    if 'finetuned' in pt_models and finetuned_path is not None:
        method_configs['finetuned'] = {'model_path': finetuned_path}
    if trt_model is not None:
        method_configs['finetuned_trt'] = {
            'engine_dir': args.trt_dir,
            'input_size': f'{trt_h}x{trt_w}',
            'precision':  'FP16',
        }

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

    # ── depth-vs-distance plot ────────────────────────────────────────────────
    plot_colors = {m: METHODS[m]['color'] for m in active_methods if m in METHODS}
    plot_labels = {
        GT_NAME:         'Zivid GT (spatial spread)',
        RS_NAME:         METHODS[RS_NAME]['label'],
        'original':      METHODS['original']['label'],
        'finetuned':     METHODS['finetuned']['label'],
        'finetuned_trt': METHODS['finetuned_trt']['label'],
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
