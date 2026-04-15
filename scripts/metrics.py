"""Depth quality metric dataclasses and computation helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np


# Distance bins used throughout the report
DIST_BINS: List[Tuple[float, float]] = [
    (0.05, 0.25),
    (0.25, 0.5),
    (0.5, 0.75),
    (0.75, 1.0),
    (1.0, 1.5),
    (1.5, 2.0),
]
BIN_LABELS = ["0.05–0.25 m", "0.25–0.5 m", "0.5–0.75 m", "0.75–1 m", "1–1.5 m", "1.5–2 m"]
BIN_CENTERS = [0.125, 0.375, 0.625, 0.875, 1.25, 1.75]
CLOSE_RANGE_THRESHOLD_M = 0.05


@dataclass
class FrameMetrics:
    """Quality metrics for a single (method, frame) pair vs ground truth."""
    method_name: str
    mae: float           # mean absolute error (m) — valid-only
    rmse: float          # root mean squared error (m) — valid-only
    mre: float           # mean relative error (dimensionless) — valid-only
    delta1: float        # % pixels within 1.25× of GT — valid-only
    coverage: float      # % valid pixels (pred > 0 AND gt > 0)
    time_ms: float       # wall-clock processing time
    # Penalised metrics: missing pixels (pred=0 where GT>0) count as 100% error
    mae_pen: float = 0.0
    mre_pen: float = 0.0


@dataclass
class AggregateStats:
    """Per-method aggregate statistics over all benchmark frames."""
    method_name: str
    label: str
    color: str
    mae_mean: float
    mae_std: float
    rmse_mean: float
    mre_mean: float
    delta1_mean: float
    coverage_mean: float
    fps_mean: float
    time_ms_mean: float
    time_ms_std: float
    close_range_coverage: float   # % valid where GT < CLOSE_RANGE_THRESHOLD_M
    mae_pen_mean: float = 0.0    # penalised MAE (holes count as full error)
    mre_pen_mean: float = 0.0    # penalised MRE (holes count as 100% error)
    gpu_load_mean: float = 0.0   # GPU utilisation % during inference


@dataclass
class BenchmarkResults:
    """All data collected during a benchmark run."""
    method_names: List[str]
    method_labels: Dict[str, str]
    method_colors: Dict[str, str]
    ground_truth_name: str
    n_frames: int
    width: int
    height: int
    all_metrics: List[FrameMetrics]
    viz_frames: List[Dict[str, np.ndarray]]    # [{method_name: float32_m_array}, ...]
    coverage_maps: Dict[str, np.ndarray]        # method_name -> (H,W) float [0,1]
    dist_bin_mae: Dict[str, List[List[float]]] # method_name -> [frame][bin]
    close_range_valid: Dict[str, List[float]]  # method_name -> per-frame %
    source: str
    method_configs: Dict[str, Dict[str, str]] = field(default_factory=dict)  # method_name -> config dict


def compute_metrics(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
    elapsed_ms: float,
    method_name: str,
) -> FrameMetrics:
    """Compute per-frame quality metrics between prediction and ground truth.

    Two sets of error metrics are computed:
      - **valid-only** (mae, mre): only pixels where both pred and GT > 0.
      - **penalised** (mae_pen, mre_pen): over all GT > 0 pixels.  Where
        pred == 0 (hole), the error equals the GT depth itself (i.e. 100%
        relative error).  This makes methods with poor coverage pay a price
        in the error scores, not just in coverage %.
    """
    gt_mask = gt_m > 0.0
    valid = gt_mask & (pred_m > 0.0)
    n_gt = int(gt_mask.sum())
    n_valid = int(valid.sum())
    n_total = gt_m.size

    if n_valid == 0:
        return FrameMetrics(method_name, 0.0, 0.0, 0.0, 0.0, 0.0, elapsed_ms)

    # --- valid-only metrics (unchanged) ---
    p, g = pred_m[valid], gt_m[valid]
    diff = np.abs(p - g)
    mae = float(diff.mean())
    rmse = float(np.sqrt((diff ** 2).mean()))
    mre = float((diff / (g + 1e-6)).mean())
    ratio = np.maximum(p / (g + 1e-6), g / (p + 1e-6))
    delta1 = float((ratio < 1.25).mean()) * 100.0
    coverage = float(n_valid / n_total) * 100.0

    # --- penalised metrics (missing pixels = full GT depth as error) ---
    if n_gt > 0:
        missing = gt_mask & (pred_m <= 0.0)
        # For missing pixels: absolute error = gt depth, relative error = 1.0
        pen_abs = np.zeros_like(gt_m)
        pen_abs[valid] = diff
        pen_abs[missing] = gt_m[missing]
        mae_pen = float(pen_abs[gt_mask].mean())

        pen_rel = np.zeros_like(gt_m)
        pen_rel[valid] = diff / (gt_m[valid] + 1e-6)
        pen_rel[missing] = 1.0  # 100% relative error for holes
        mre_pen = float(pen_rel[gt_mask].mean())
    else:
        mae_pen = mae
        mre_pen = mre

    return FrameMetrics(method_name, mae, rmse, mre, delta1, coverage, elapsed_ms,
                        mae_pen=mae_pen, mre_pen=mre_pen)


def compute_bin_mae(
    pred_m: np.ndarray,
    gt_m: np.ndarray,
) -> List[float]:
    """MAE per distance bin; returns NaN for bins with no valid GT pixels."""
    result = []
    for lo, hi in DIST_BINS:
        mask = (gt_m >= lo) & (gt_m < hi) & (gt_m > 0) & (pred_m > 0)
        if mask.sum() == 0:
            result.append(float("nan"))
        else:
            result.append(float(np.abs(pred_m[mask] - gt_m[mask]).mean()))
    return result


def aggregate(
    results: BenchmarkResults,
    timing_ms: Dict[str, float],
    gpu_load: Optional[Dict[str, float]] = None,
) -> Dict[str, AggregateStats]:
    """Compute per-method aggregate statistics from raw benchmark results."""
    stats: Dict[str, AggregateStats] = {}
    gt = results.ground_truth_name

    for name in results.method_names:
        label = results.method_labels.get(name, name)
        color = results.method_colors.get(name, "#888888")
        fm_list = [m for m in results.all_metrics if m.method_name == name]
        if not fm_list:
            continue

        t_ms = timing_ms.get(name, 0.0)
        if name == "hardware":
            fps = 30.0
            t_ms = 1000.0 / 30.0
        else:
            fps = 1000.0 / t_ms if t_ms > 0 else float("inf")

        cr_vals = results.close_range_valid.get(name, [0.0])

        stats[name] = AggregateStats(
            method_name=name,
            label=label,
            color=color,
            mae_mean=float(np.mean([m.mae for m in fm_list])),
            mae_std=float(np.std([m.mae for m in fm_list])),
            rmse_mean=float(np.mean([m.rmse for m in fm_list])),
            mre_mean=float(np.mean([m.mre for m in fm_list])),
            delta1_mean=float(np.mean([m.delta1 for m in fm_list])),
            coverage_mean=float(np.mean([m.coverage for m in fm_list])),
            fps_mean=fps,
            time_ms_mean=t_ms,
            time_ms_std=0.0,
            close_range_coverage=float(np.mean(cr_vals)) if cr_vals else 0.0,
            mae_pen_mean=float(np.mean([m.mae_pen for m in fm_list])),
            mre_pen_mean=float(np.mean([m.mre_pen for m in fm_list])),
            gpu_load_mean=(gpu_load or {}).get(name, 0.0),
        )
    return stats