"""Benchmark original vs fine-tuned FastFoundationStereo on the Inbolt dataset.

Loads both models, runs inference on all Inbolt samples, computes depth quality
metrics against Zivid scanner ground truth, and produces an HTML report.

Also generates depth accuracy and noise plots comparing RealSense stereo predictions
(model output) against Zivid ground-truth depth across distance bins.

Usage:
  cd /home/adiroha/repos/Fast-FoundationStereo
  python scripts/benchmark_inbolt.py [--out_dir reports/inbolt_benchmark]
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from core.utils.utils import InputPadder
import Utils as U

from scripts.data_manager_inbolt import DataSource

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

#DATA_DIR         = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260322T091926Z-1-001/Data Collection' 
DATA_DIR         = r'/mnt/algonas/Local/Data/new_depth_stereo_datasets/Inbolt_datasets/Data Collection-20260415T084601Z-3-001/Data Collection'  # local path to the dataset
MODEL_PATH      = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
FINETUNED_PATH  = f'{code_dir}/../weights/20-30-48/model_finetuned_inbolt-20260415.pth'
MODEL_PATH      = f'{code_dir}/../weights/23-36-37/model_best_bp2_serialize.pth'
FINETUNED_PATH  = f'{code_dir}/../weights/23-36-37/model_finetuned_inbolt-20260415.pth'
DEFAULT_OUT     = f'{code_dir}/../reports/inbolt_benchmark'
#FARO_DIR       = r'data/faro'  # local path to FARO dataset --- IGNORE ---
# ORIGINAL_PATH  = f'{code_dir}/../weights/20-30-48/model_best_bp2_serialize.pth'
# FINETUNED_PATH = f'{code_dir}/../weights/20-30-48/model_finetuned_faro.pth'
# DEFAULT_OUT    = f'{code_dir}/../reports/faro_benchmark'

BF              = 50.102706998586 * 385.509887695312  #49470.45   # focal_px * baseline_mm  (calibrated from camera)
ITERS           = 8          # GRU iterations
N_VIZ           = 5         # number of frames saved for visual comparison in report

DEPTH_BIN_SIZE_M = 0.1       # width of each distance bin for accuracy/noise plots
MAX_DEPTH_M      = 2.0       # maximum depth considered in plots

METHODS = {
    "original":  {"label": "Original model",    "color": "#2980b9"},
    "finetuned": {"label": "Fine-tuned on INBOLT", "color": "#e74c3c"},
    "inbolt_gt":   {"label": "INBOLT GT",            "color": "#27ae60"},
}
GT_NAME = "inbolt_gt"


# ── depth-vs-distance analysis ────────────────────────────────────────────────

class DepthBinAccumulator:
    """Accumulates mean and std-dev of depth values per GT-distance bin.

    Bins are defined by Zivid GT depth, so *every* pixel whose GT depth falls in
    [k * bin_size, (k+1) * bin_size) contributes to bin k.  Works for any sensor
    (model predictions OR Zivid GT values themselves — the latter gives an
    estimate of within-bin spatial variation / measurement noise).
    """

    def __init__(self, bin_size_m: float = DEPTH_BIN_SIZE_M, max_depth_m: float = MAX_DEPTH_M):
        self.bin_size   = bin_size_m
        self.n_bins     = int(np.ceil(max_depth_m / bin_size_m))
        self.count      = np.zeros(self.n_bins, dtype=np.float64)
        self.sum_       = np.zeros(self.n_bins, dtype=np.float64)
        self.sum_sq     = np.zeros(self.n_bins, dtype=np.float64)

    def update(self, values_m: np.ndarray, gt_m: np.ndarray):
        """Add one frame of data.

        Parameters
        ----------
        values_m : (H, W) array of the depth values to accumulate (model or GT).
        gt_m     : (H, W) array of Zivid GT depths that define which bin each pixel falls in.
        """
        valid = (gt_m > 0) & (values_m > 0)
        if not valid.any():
            return
        v_vals = values_m[valid].ravel().astype(np.float64)
        v_gt   = gt_m[valid].ravel().astype(np.float64)

        bins = np.floor(v_gt / self.bin_size).astype(np.int32)
        bins = np.clip(bins, 0, self.n_bins - 1)

        np.add.at(self.count,  bins, 1.0)
        np.add.at(self.sum_,   bins, v_vals)
        np.add.at(self.sum_sq, bins, v_vals ** 2)

    @property
    def bin_centers(self) -> np.ndarray:
        return (np.arange(self.n_bins) + 0.5) * self.bin_size

    def mean(self) -> np.ndarray:
        c = np.maximum(self.count, 1)
        return np.where(self.count > 0, self.sum_ / c, np.nan)

    def std(self) -> np.ndarray:
        c = np.maximum(self.count, 1)
        m = np.where(self.count > 0, self.sum_ / c, np.nan)
        v = np.where(self.count > 0, self.sum_sq / c - m ** 2, np.nan)
        return np.sqrt(np.maximum(v, 0.0))

def plot_depth_vs_distance(
    accumulators: dict,          # {label: DepthBinAccumulator}
    colors: dict,                # {label: color_str}
    out_path: Path,
    min_count: int = 100,        # bins with fewer samples are hidden
):
    """Produce a two-panel figure:
      left  – actual depth (Zivid GT) vs measured/predicted depth
      right – noise (std-dev of measured depth) vs actual depth
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for label, acc in accumulators.items():
        x     = acc.bin_centers
        mu    = acc.mean()
        sigma = acc.std()
        color = colors.get(label, None)

        valid = acc.count >= min_count
        xv, muv, sv = x[valid], mu[valid], sigma[valid]

        ax0 = axes[0]
        ax0.plot(xv, muv, label=label, color=color, linewidth=1.5)
        ax0.fill_between(xv, muv - sv, muv + sv, alpha=0.15, color=color)

        axes[1].plot(xv, sv * 1000, label=label, color=color, linewidth=1.5)

    # ideal line
    lim = MAX_DEPTH_M
    axes[0].plot([0, lim], [0, lim], 'k--', linewidth=1, label='ideal (y = x)')
    axes[0].set_xlim(0, lim)
    axes[0].set_ylim(0, lim)
    axes[0].set_xlabel('Actual depth — Zivid GT (m)')
    axes[0].set_ylabel('Measured depth (m)')
    axes[0].set_title('Depth Accuracy: Actual vs Measured\n(shaded band = ±1 std dev)')
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.4)

    axes[1].set_xlim(0, lim)
    axes[1].set_xlabel('Actual depth — Zivid GT (m)')
    axes[1].set_ylabel('Noise / Std Dev (mm)')
    axes[1].set_title('Depth Noise per Distance Bin')
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    logging.info(f"Saved depth-vs-distance plot → {out_path}")

def _to_1d_float_array(values, name: str) -> np.ndarray:
    """Convert *values* to a finite 1D float array."""
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        raise ValueError(f"{name} must contain at least one value")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values")
    return arr

def fit_depth_scale_regression(
    gt_delta_mm,
    measured_delta_mm,
    fit_intercept: bool = False,
) -> dict:
    """Fit a linear depth-scale regression and compute residual statistics.

    Parameters
    ----------
    gt_delta_mm : array-like
        Ground-truth floor/depth deltas in millimetres.
    measured_delta_mm : array-like
        Measured deltas from one sensor/model in millimetres.
    fit_intercept : bool, default=False
        If False, uses a through-origin fit `y = slope * x`, which matches the
        style of the attached plot.  If True, fits `y = slope * x + intercept`.

    Returns
    -------
    dict
        Contains slope, intercept, fitted values, residuals, RMSE, and masks.
    """
    x = _to_1d_float_array(gt_delta_mm, "gt_delta_mm")
    y = _to_1d_float_array(measured_delta_mm, "measured_delta_mm")

    if x.shape != y.shape:
        raise ValueError("gt_delta_mm and measured_delta_mm must have the same shape")
    if x.size < 2:
        raise ValueError("At least two samples are required for regression")

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if x.size < 2:
        raise ValueError("Need at least two finite samples after filtering")

    if fit_intercept:
        slope, intercept = np.polyfit(x, y, deg=1)
    else:
        denom = float(np.dot(x, x))
        if denom <= 0:
            raise ValueError("Cannot fit a through-origin regression when gt deltas are all zero")
        slope = float(np.dot(x, y) / denom)
        intercept = 0.0

    fitted = slope * x + intercept
    residuals = y - fitted
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    return {
        "gt_delta_mm": x,
        "measured_delta_mm": y,
        "slope": float(slope),
        "intercept": float(intercept),
        "fitted_mm": fitted,
        "residuals_mm": residuals,
        "rmse_mm": rmse,
        "fit_intercept": fit_intercept,
    }

def build_example_depth_scale_regression_series(gt_delta_mm, rs_delta_mm, zv_delta_mm) -> dict:
    """Return example depth-delta series that reproduces the attached figure.

    The values approximate the plot shown in the screenshot:
      - RealSense has a noticeable scale bias.
      - Zivid stays close to the ideal slope of 1.
    """
    gt_delta_mm = np.array([0, 100, 200, 300, 400, 500, 600, 700], dtype=np.float64) if gt_delta_mm is None else gt_delta_mm
    rs_delta_mm = np.array([0.0, 104.0, 218.0, 323.0, 433.0, 542.0, 664.0, 754.0], dtype=np.float64) if rs_delta_mm is None else rs_delta_mm
    zv_delta_mm = np.array([0.0, 101.0, 201.0, 301.0, 401.0, 502.0, 602.0, 707.0], dtype=np.float64) if zv_delta_mm is None else zv_delta_mm

    return {
        "realsense": {
            "gt_delta_mm": gt_delta_mm,
            "measured_delta_mm": rs_delta_mm,
            "color": "#e74c3c",
            "marker": "s",
            "label": "realsense",
        },
        "zivid": {
            "gt_delta_mm": gt_delta_mm,
            "measured_delta_mm": zv_delta_mm,
            "color": "#2980b9",
            "marker": "o",
            "label": "zivid",
        },
    }

def plot_depth_scale_regression(
    series_map: dict,
    out_path: Path,
    title: str = "Depth Scale Regression — dataset_depth_bias",
    fit_intercept: bool = False,
    ideal_slope: float = 1.0,
):
    """Create the two-panel regression + residuals figure from paired series.

    Parameters
    ----------
    series_map : dict
        Mapping of series name to configuration dict. Each entry should provide:
          - gt_delta_mm
          - measured_delta_mm
        and may optionally include:
          - label
          - color
          - marker
    out_path : Path
        Destination PNG path.
    title : str
        Figure title.
    fit_intercept : bool
        Whether to fit a free intercept. Defaults to a through-origin fit.
    ideal_slope : float
        Slope of the ideal reference line shown on the left panel.
    """
    if not series_map:
        raise ValueError("series_map must contain at least one series")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fit_results = []
    max_x = 0.0
    max_y = 0.0

    for default_name, cfg in series_map.items():
        result = fit_depth_scale_regression(
            gt_delta_mm=cfg["gt_delta_mm"],
            measured_delta_mm=cfg["measured_delta_mm"],
            fit_intercept=fit_intercept,
        )
        result["label"] = cfg.get("label", default_name)
        result["color"] = cfg.get("color", None)
        result["marker"] = cfg.get("marker", "o")
        fit_results.append(result)
        max_x = max(max_x, float(np.max(result["gt_delta_mm"])))
        max_y = max(max_y, float(np.max(result["measured_delta_mm"])))

    lim = max(max_x, max_y)
    fit_x = np.linspace(0.0, lim, 200)

    for result in fit_results:
        label = result["label"]
        color = result["color"]
        marker = result["marker"]
        x = result["gt_delta_mm"]
        y = result["measured_delta_mm"]
        slope = result["slope"]
        intercept = result["intercept"]
        rmse = result["rmse_mm"]

        axes[0].scatter(x, y, color=color, marker=marker, s=70, label=f"{label} (raw)", zorder=3)
        axes[0].plot(
            fit_x,
            slope * fit_x + intercept,
            color=color,
            linewidth=2.0,
            label=(
                f"{label} fit: slope={slope:.3f}, intercept={intercept:.1f}mm, RMSE={rmse:.1f}mm"
                if fit_intercept else
                f"{label} fit: slope={slope:.3f}, RMSE={rmse:.1f}mm"
            ),
        )

        axes[1].scatter(
            x,
            result["residuals_mm"],
            color=color,
            marker=marker,
            s=70,
            label=f"{label} (RMSE={rmse:.1f}mm)",
            zorder=3,
        )

    axes[0].plot(
        fit_x,
        ideal_slope * fit_x,
        linestyle="--",
        color="gray",
        linewidth=1.5,
        label=f"ideal (slope={ideal_slope:.1f})",
    )
    axes[0].set_xlabel("Ground Truth Delta (mm)")
    axes[0].set_ylabel("Measured Depth Delta (mm)")
    axes[0].set_title("Floor Depth Delta: Measured vs Ground Truth")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=9, loc="upper left")

    axes[1].axhline(0.0, linestyle="--", color="gray", linewidth=1.2)
    axes[1].set_xlabel("Ground Truth Delta (mm)")
    axes[1].set_ylabel("Residual (mm)")
    axes[1].set_title("Residuals (Measured − Fit)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=9, loc="upper left")

    axes[0].set_xlim(-0.05 * max(lim, 1.0), lim * 1.05)
    axes[1].set_xlim(-0.05 * max(lim, 1.0), lim * 1.05)

    residual_values = np.concatenate([r["residuals_mm"] for r in fit_results])
    residual_abs_max = max(1.0, float(np.max(np.abs(residual_values))))
    axes[1].set_ylim(-residual_abs_max * 1.15, residual_abs_max * 1.15)

    fig.suptitle(title, fontsize=18, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Saved depth-scale regression plot → {out_path}")


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

@torch.no_grad()
def infer_depth_nobf_m(model, left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Run stereo inference on an IR pair that returns depth; return depth map in metres (H×W float32)."""
    left_t, right_t = _preprocess_ir(left, right)
    padder = InputPadder(left_t.shape, divis_by=32, force_square=False)
    left_t, right_t = padder.pad(left_t, right_t)

    with torch.amp.autocast('cuda', enabled=True, dtype=U.AMP_DTYPE):
        depth = model.forward(left_t, right_t, iters=ITERS, test_mode=True)

    depth = padder.unpad(depth.float())
    depth_np = depth.cpu().numpy().reshape(left.shape[:2]).clip(0, None)

    depth_m = np.zeros_like(depth_np)
    valid = depth_np > 0
    depth_m[valid] = depth_np[valid] / 1000.0   # mm → m
    return depth_m


# ── inbolt graphs ─────────────────────────────────────────────────────────────────────

def main_inbolt_graphs():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', default=DEFAULT_OUT, help='Output directory for the report')
    parser.add_argument('--data_dir', default=DATA_DIR, help='Path to dataset root')
    parser.add_argument('--original', default=MODEL_PATH, help='Path to original model weights')
    parser.add_argument('--finetuned', default=FINETUNED_PATH, help='Path to fine-tuned model weights')
    parser.add_argument('--n_viz', type=int, default=N_VIZ, help='Frames saved for visual comparison')
    args = parser.parse_args()

    U.set_logging_format()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── dataset ───────────────────────────────────────────────────────────────
    source = DataSource()
    n = source.init_directory(input_rectified=args.data_dir)
    logging.info(f"Found {n} samples in {args.data_dir}")
    if n == 0:
        logging.error("No samples found — check DATA_DIR path")
        return


    #import cv2 as _cv2   # local import to avoid top-level dependency if already imported
    gt_depth_diff = np.arange(n)*100 # mm
    rs_depth_diff = np.arange(n)*0 # mm
    zv_depth_diff = np.arange(n)*0 # zivid mm
    rs_ref = None
    zv_ref = None
    for idx in range(n):
        data  = source.get_item(idx)
        left  = data['left']
        right = data['right']
        zv_mm = data['depth_zivid'].astype(np.float32)   # Zivid GT in mm
        rs_mm = data['depth_rs'].astype(np.float32)   # RealSense depth in mm

        # # Resize Zivid depth to match RealSense IR image resolution for pixel-level comparison
        # rs_h, rs_w = left.shape[:2]
        # if gt_mm.shape != (rs_h, rs_w):
        #     #gt_mm = _cv2.resize(gt_mm, (rs_w, rs_h), interpolation=_cv2.INTER_NEAREST)
        #     print(f"Shape mismatch: gt_mm {gt_mm.shape} vs rs {rs_h, rs_w}")
        rs_valid           = (rs_mm > rs_mm.max()*0.8) 
        zv_valid           = (zv_mm > zv_mm.max()*0.8) 
        if idx == 0:
            rs_ref = np.nanmean(rs_mm[rs_valid])
            zv_ref = np.nanmean(zv_mm[zv_valid])
        else:
            rs_depth_diff[idx] = np.nanmean(rs_mm[rs_valid]) - rs_ref
            zv_depth_diff[idx] = np.nanmean(zv_mm[zv_valid]) - zv_ref


    sm = build_example_depth_scale_regression_series(gt_depth_diff, rs_depth_diff, zv_depth_diff)
    plot_depth_scale_regression(sm, out_path=Path(DEFAULT_OUT) / "depth_scale_comparison.png", title="Depth Scale Comparison")

    logging.info(f"All outputs written to {out_dir}")

# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out_dir', default=DEFAULT_OUT, help='Output directory for the report')
    parser.add_argument('--data_dir', default=DATA_DIR, help='Path to dataset root')
    parser.add_argument('--original', default=MODEL_PATH, help='Path to original model weights')
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
    n = source.init_directory(input_rectified=args.data_dir)
    logging.info(f"Found {n} samples in {args.data_dir}")
    if n == 0:
        logging.error("No samples found — check DATA_DIR path")
        return

    # ── accumulators ──────────────────────────────────────────────────────────
    all_metrics       = []
    viz_frames        = []
    valid_acc         = {}     # will be init on first frame
    dist_bin_mae      = {m: [] for m in active_methods}
    close_range_valid = {m: [] for m in active_methods}
    timing_ms_raw     = {m: [] for m in models}
    H = W = None

    # depth-vs-distance accumulators:
    #   "zivid_gt" – Zivid depth values binned by Zivid GT (shows intra-bin spatial spread)
    #   one entry per stereo model – model predictions binned by Zivid GT
    depth_acc_keys = ["zivid_gt"] + list(models.keys())
    depth_accs = {k: DepthBinAccumulator() for k in depth_acc_keys}

    #import cv2 as _cv2   # local import to avoid top-level dependency if already imported

    for idx in range(n):
        data  = source.get_item(idx)
        left  = data['left']
        right = data['right']
        gt_mm = data['depth_zivid'].astype(np.float32)   # Zivid GT in mm
        rs_mm = data['depth_rs'].astype(np.float32)   # RealSense depth in mm

        # Resize Zivid depth to match RealSense IR image resolution for pixel-level comparison
        rs_h, rs_w = left.shape[:2]
        if gt_mm.shape != (rs_h, rs_w):
            #gt_mm = _cv2.resize(gt_mm, (rs_w, rs_h), interpolation=_cv2.INTER_NEAREST)
            print(f"Shape mismatch: gt_mm {gt_mm.shape} vs rs {rs_h, rs_w}")

        if H is None:
            H, W = rs_h, rs_w
            for m in active_methods:
                valid_acc[m] = np.zeros((H, W), np.float32)

        gt_m = gt_mm / 1000.0   # mm → m
        rs_m = rs_mm / 1000.0   # mm → m

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

        # depth-vs-distance: accumulate per-bin stats
        depth_accs["zivid_gt"].update(gt_m, gt_m)   # GT vs itself → intra-bin spread
        for mname in models:
            depth_accs[mname].update(frame_depths[mname], gt_m)

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
        source=f"INBOLT dataset ({args.data_dir})",
        method_configs={
            "original":  {"model_path": args.original},
            "finetuned": {"model_path": args.finetuned},
        },
    )

    stats = aggregate(results, mean_timing)

    # ── generate report ───────────────────────────────────────────────────────
    reporter = ReportGenerator(results, stats, out_dir)
    reporter.generate()

    # ── depth-vs-distance comparison plot ────────────────────────────────────
    plot_colors = {
        "zivid_gt": METHODS[GT_NAME]["color"],
        **{m: METHODS[m]["color"] for m in models if m in METHODS},
    }
    plot_labels = {
        "zivid_gt":  "Zivid GT (spatial spread)",
        "original":  METHODS["original"]["label"],
        "finetuned": METHODS["finetuned"]["label"],
    }
    # rename keys to human-readable labels for the plot
    labeled_accs = {plot_labels.get(k, k): v
                    for k, v in depth_accs.items()
                    if depth_accs[k].count.sum() > 0}
    labeled_colors = {plot_labels.get(k, k): plot_colors.get(k)
                      for k in depth_accs if depth_accs[k].count.sum() > 0}

    plot_depth_vs_distance(
        accumulators=labeled_accs,
        colors=labeled_colors,
        out_path=out_dir / "depth_vs_distance.png",
    )
    logging.info(f"All outputs written to {out_dir}")


if __name__ == '__main__':
    # 1. works
    #sm = build_example_depth_scale_regression_series()
    #plot_depth_scale_regression(sm, out_path=Path(DEFAULT_OUT) / "depth_scale_regression_example.png", title="Example Depth Scale Regression")

    # 2. inbolt data
    main_inbolt_graphs()

    # 3. full benchmark + report
    #main()
