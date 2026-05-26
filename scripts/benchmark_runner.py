"""BenchmarkRunner — orchestrates warm-up, frame collection and metric accumulation."""

from __future__ import annotations

import time
from typing import Dict, List, Optional

import numpy as np

from .metrics import (
    CLOSE_RANGE_THRESHOLD_M,
    DIST_BINS,
    BenchmarkResults,
    FrameMetrics,
    compute_bin_mae,
    compute_metrics,
)
from .methods import DepthMethod


_GPU_LOAD_PATH = None

def _read_gpu_load() -> float:
    """Read GPU utilisation from sysfs (Jetson Orin). Returns 0-100 or -1 on failure."""
    global _GPU_LOAD_PATH
    if _GPU_LOAD_PATH is None:
        import glob
        candidates = glob.glob("/sys/devices/platform/bus@0/*/load") + \
                     glob.glob("/sys/devices/platform/gpu*/load")
        for c in candidates:
            if "gpu" in c.lower() or "17000000" in c:
                _GPU_LOAD_PATH = c
                break
        if _GPU_LOAD_PATH is None:
            _GPU_LOAD_PATH = ""
    if not _GPU_LOAD_PATH:
        return -1.0
    try:
        with open(_GPU_LOAD_PATH) as f:
            return int(f.read().strip()) / 10.0
    except Exception:
        return -1.0


class BenchmarkRunner:
    """Runs the benchmark loop and collects all data needed by ReportGenerator.

    Usage::

        runner = BenchmarkRunner(capture, methods, ground_truth_name="nn_accurate")
        results = runner.run(n_frames=30, warmup=5, n_viz=3)
        timing  = runner.measure_timing(n_frames=5)
    """

    def __init__(
        self,
        capture,
        methods: List[DepthMethod],
        ground_truth_name: str,
    ) -> None:
        self._capture = capture
        self._methods = methods
        self._available = [m for m in methods if m.available]
        self._gt_name = ground_truth_name
        # Mark ground-truth flag on each method
        for m in self._available:
            m.is_ground_truth = m.name == ground_truth_name

    # ------------------------------------------------------------------ public

    def run(
        self,
        n_frames: int = 30,
        warmup: int = 5,
        n_viz: int = 3,
    ) -> BenchmarkResults:
        W = getattr(self._capture, "_width", None) or getattr(self._capture, "_w", 640)
        H = getattr(self._capture, "_height", None) or getattr(self._capture, "_h", 480)
        n_viz = min(n_viz, n_frames)

        self._print_plan()
        self._warmup(warmup)

        all_metrics: List[FrameMetrics] = []
        viz_frames: List[Dict] = []
        valid_acc = {m.name: np.zeros((H, W), np.float32) for m in self._available}
        dist_bin_mae: Dict[str, List[List[float]]] = {m.name: [] for m in self._available}
        close_range_valid: Dict[str, List[float]] = {m.name: [] for m in self._available}

        print(f"\nRunning benchmark ({n_frames} frames)...")
        for frame_idx in range(n_frames):
            frame_depths = self._process_frame()

            gt_m = frame_depths.get(self._gt_name, np.zeros((H, W), np.float32))
            self._accumulate(
                frame_idx, frame_depths, gt_m, H, W,
                all_metrics, valid_acc, dist_bin_mae, close_range_valid,
            )
            if frame_idx < n_viz:
                viz_frames.append({k: v.copy() for k, v in frame_depths.items()})
            if (frame_idx + 1) % 10 == 0 or (frame_idx + 1) == n_frames:
                print(f"  Frame {frame_idx + 1}/{n_frames}")

        # Normalise coverage maps to [0, 1]
        for name in valid_acc:
            valid_acc[name] /= max(n_frames, 1)

        return BenchmarkResults(
            method_names=[m.name for m in self._available],
            method_labels={m.name: m.label for m in self._methods},
            method_colors={m.name: m.color for m in self._methods},
            ground_truth_name=self._gt_name,
            n_frames=n_frames,
            width=W,
            height=H,
            all_metrics=all_metrics,
            viz_frames=viz_frames,
            coverage_maps=valid_acc,
            dist_bin_mae=dist_bin_mae,
            close_range_valid=close_range_valid,
            source=self._source_label(),
            method_configs={m.name: m.config for m in self._available if m.config},
        )

    def measure_timing(self, n_frames: int = 5) -> Dict[str, float]:
        """Return mean processing time (ms) per available method."""
        times: Dict[str, List[float]] = {m.name: [] for m in self._available}
        for _ in range(n_frames):
            fs = self._capture.get_frames()
            for m in self._available:
                t0 = time.monotonic()
                try:
                    m.process_fn(fs)
                except Exception:
                    pass
                times[m.name].append((time.monotonic() - t0) * 1000.0)
        return {name: float(np.mean(ts)) if ts else 0.0
                for name, ts in times.items()}

    def measure_gpu_load(
        self, duration_sec: float = 60.0, cooldown_sec: float = 60.0,
    ) -> Dict[str, float]:
        """Measure GPU load per method in isolation (one method at a time).

        For each method, idles for *cooldown_sec* to let the GPU settle, then
        runs the method for *duration_sec* while a background thread samples
        GPU utilisation every 50 ms.  This gives a realistic sustained GPU
        load, not a peak snapshot.
        """
        import threading

        gpu_loads: Dict[str, float] = {}

        for i, m in enumerate(self._available):
            # Cooldown — let GPU idle between methods
            if cooldown_sec > 0 and i > 0:
                print(f"  Cooldown {cooldown_sec:.0f}s...")
                time.sleep(cooldown_sec)

            # Baseline reading before inference starts
            baseline = _read_gpu_load()

            samples: List[float] = []
            stop_event = threading.Event()

            def _sampler():
                while not stop_event.is_set():
                    val = _read_gpu_load()
                    if val >= 0:
                        samples.append(val)
                    stop_event.wait(0.05)

            # Start sampling
            t = threading.Thread(target=_sampler, daemon=True)
            t.start()

            # Run method in isolation for duration_sec
            t_end = time.monotonic() + duration_sec
            n = 0
            while time.monotonic() < t_end:
                fs = self._capture.get_frames()
                try:
                    m.process_fn(fs)
                except Exception:
                    pass
                n += 1

            # Stop sampling
            stop_event.set()
            t.join(timeout=1.0)

            gpu_loads[m.name] = float(np.mean(samples)) if samples else 0.0
            print(f"  {m.label}: GPU {gpu_loads[m.name]:.0f}% "
                  f"(baseline {baseline:.0f}%, {n} frames in {duration_sec:.0f}s)")

        return gpu_loads

    # ------------------------------------------------------------------ private

    def _print_plan(self) -> None:
        print(f"Ground truth: {self._gt_name}")
        for m in self._methods:
            if m.available:
                tag = " [GT]" if m.is_ground_truth else ""
                print(f"  [OK] {m.label}{tag}")
            else:
                print(f"  [--] {m.label}  ({m.skip_reason})")

    def _warmup(self, n: int) -> None:
        if n <= 0:
            return
        print(f"\nWarming up ({n} frames)...")
        for _ in range(n):
            fs = self._capture.get_frames()
            for m in self._available:
                try:
                    m.process_fn(fs)
                except Exception:
                    pass

    def _process_frame(self) -> Dict[str, np.ndarray]:
        import cv2
        fs = self._capture.get_frames()
        W = getattr(self._capture, "_width", None) or getattr(self._capture, "_w", 640)
        H = getattr(self._capture, "_height", None) or getattr(self._capture, "_h", 480)
        result: Dict[str, np.ndarray] = {}
        for m in self._available:
            try:
                pred = m.process_fn(fs)
            except Exception:
                pred = np.zeros((H, W), dtype=np.float32)
            pred = pred.astype(np.float32) if pred.dtype != np.float32 else pred
            # Resize to camera resolution if method returns different size
            if pred.shape[:2] != (H, W):
                pred = cv2.resize(pred, (W, H))
            result[m.name] = pred
        return result

    def _accumulate(
        self,
        frame_idx: int,
        frame_depths: Dict[str, np.ndarray],
        gt_m: np.ndarray,
        H: int, W: int,
        all_metrics: List[FrameMetrics],
        valid_acc: Dict,
        dist_bin_mae: Dict,
        close_range_valid: Dict,
    ) -> None:
        gt_close_mask = (gt_m > 0) & (gt_m < CLOSE_RANGE_THRESHOLD_M)
        n_close = int(gt_close_mask.sum())

        for m in self._available:
            pred = frame_depths[m.name]
            valid_acc[m.name] += (pred > 0).astype(np.float32)

            if m.is_ground_truth:
                fm = FrameMetrics(
                    m.name, 0.0, 0.0, 0.0, 100.0,
                    float((pred > 0).mean()) * 100.0, 0.0,
                    mae_pen=0.0, mre_pen=0.0,
                )
            else:
                fm = compute_metrics(pred, gt_m, 0.0, m.name)
            all_metrics.append(fm)

            dist_bin_mae[m.name].append(compute_bin_mae(pred, gt_m))

            if n_close > 0:
                close_cov = float((pred[gt_close_mask] > 0).mean()) * 100.0
            else:
                close_cov = 0.0
            close_range_valid[m.name].append(close_cov)

    def _source_label(self) -> str:
        from .capture import SyntheticCapture
        if isinstance(self._capture, SyntheticCapture):
            return "synthetic"
        bag = getattr(self._capture, "_bag", None)
        return f"bag:{bag}" if bag else "live"