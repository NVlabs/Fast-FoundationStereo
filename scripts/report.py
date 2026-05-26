"""ReportGenerator — produces all PNG figures, HTML index, and JSON results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

# import matplotlib
# matplotlib.use("Agg")  # headless — must be before pyplot import
import matplotlib.pyplot as plt
import numpy as np

try:
    from .metrics import (
        CLOSE_RANGE_THRESHOLD_M,
        BIN_CENTERS,
        BIN_LABELS,
        AggregateStats,
        BenchmarkResults,
    )
except ImportError:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).parent))
    from metrics import (
        CLOSE_RANGE_THRESHOLD_M,
        BIN_CENTERS,
        BIN_LABELS,
        AggregateStats,
        BenchmarkResults,
    )


class ReportGenerator:
    """Generates the full depth quality report in *output_dir*.

    Output structure::

        output_dir/
          index.html               ← self-contained HTML (relative img srcs)
          results.json             ← aggregate stats
          depth_comparison.png     ← side-by-side depth maps
          error_maps.png           ← |pred − GT| per method
          coverage_heatmaps.png    ← valid-pixel fraction over all frames
          distance_error_curve.png ← MAE vs distance bins
          error_histograms.png     ← per-pixel error distributions
          summary_table.png        ← statistics table
          close_range_analysis.png ← < 0.55 m coverage bar + per-frame curve
          timing_bars.png          ← FPS per method
    """

    def __init__(
        self,
        results: BenchmarkResults,
        stats: Dict[str, AggregateStats],
        output_dir: Path,
    ) -> None:
        self._r = results
        self._stats = stats
        self._out = Path(output_dir)
        self._out.mkdir(parents=True, exist_ok=True)
        self._gt = results.ground_truth_name
        self._non_gt = [n for n in results.method_names if n != self._gt]

    def generate(self) -> None:
        """Generate all figures, JSON, and HTML index."""
        fig_paths = [
            self._fig_depth_comparison(),
            self._fig_error_maps(),
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

    # ------------------------------------------------------------------ figures

    def _save(self, fig, name: str) -> str:
        fig.savefig(self._out / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return name

    def _empty_fig(self, filename: str, msg: str) -> str:
        fig, ax = plt.subplots(figsize=(6, 3), dpi=100)
        ax.axis("off")
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color="gray")
        return self._save(fig, filename)

    def _depth_cmap(self):
        cmap = plt.get_cmap("plasma").copy()
        cmap.set_under("black")
        return cmap

    def _grid_layout(self, n, max_cols=4):
        """Return (nrows, ncols) for a grid that fits n items with at most max_cols columns."""
        ncols = min(n, max_cols)
        nrows = (n + ncols - 1) // ncols
        return nrows, ncols

    def _fig_depth_comparison(self) -> str:
        if not self._r.viz_frames:
            return self._empty_fig("depth_comparison.png", "No viz frames")
        vf = self._r.viz_frames[0]
        names = [n for n in self._r.method_names if n in vf]
        n = len(names)
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.asarray(axes).flatten().tolist()
        cmap = self._depth_cmap()
        for i, (ax, name) in enumerate(zip(axes[:n], names)):
            im = ax.imshow(vf[name], cmap=cmap, vmin=1e-4, vmax=5.0)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="meters")
            ax.set_title(self._r.method_labels.get(name, name), fontsize=9, wrap=True)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Depth Map Comparison (single frame)", fontsize=11, y=1.02)
        fig.tight_layout()
        return self._save(fig, "depth_comparison.png")

    def _fig_error_maps(self) -> str:
        if not self._r.viz_frames or not self._non_gt:
            return self._empty_fig("error_maps.png", "No comparison methods")
        vf = self._r.viz_frames[0]
        gt = vf.get(self._gt)
        # Include GT itself first (MAE=0 sanity check), then all other methods
        names = ([self._gt] if self._gt in vf else []) + [n for n in self._non_gt if n in vf]
        if gt is None or not names:
            return self._empty_fig("error_maps.png", "Ground truth not available in viz frame")
        n = len(names)
        nrows, ncols = self._grid_layout(n)
        cmap = plt.get_cmap("hot").copy()
        cmap.set_under("#222222")
        fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
        axes = np.asarray(axes).flatten().tolist()
        for ax, name in zip(axes[:n], names):
            pred = vf[name]
            valid = (gt > 0) & (pred > 0)
            err = np.where(valid, np.abs(pred - gt), 0.0).astype(np.float32)
            im = ax.imshow(err, cmap=cmap, vmin=1e-4, vmax=0.5)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="|error| (m)")
            mean_err = float(np.abs(pred[valid] - gt[valid]).mean()) if valid.any() else 0.0
            label = self._r.method_labels.get(name, name)
            ax.set_title(f"{label}\nMAE={mean_err:.3f}m", fontsize=9)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        gt_label = self._r.method_labels.get(self._gt, self._gt)
        fig.suptitle(f"Absolute Error vs {gt_label}", fontsize=11, y=1.02)
        fig.tight_layout()
        return self._save(fig, "error_maps.png")

    def _fig_coverage_heatmaps(self) -> str:
        names = list(self._r.coverage_maps.keys())
        n = len(names)
        if n == 0:
            return self._empty_fig("coverage_heatmaps.png", "No coverage data")
        nrows, ncols = self._grid_layout(n)
        fig, axes = plt.subplots(nrows, ncols, figsize=(3.5 * ncols, 3.5 * nrows))
        axes = np.asarray(axes).flatten().tolist()
        for ax, name in zip(axes[:n], names):
            cov = self._r.coverage_maps[name]
            im = ax.imshow(cov, cmap="viridis", vmin=0, vmax=1)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Valid fraction")
            mean_cov = float(cov.mean()) * 100.0
            label = self._r.method_labels.get(name, name)
            ax.set_title(f"{label}\n{mean_cov:.1f}% mean", fontsize=9)
            ax.axis("off")
        for ax in axes[n:]:
            ax.axis("off")
        fig.suptitle("Valid Pixel Coverage Heatmap (all frames avg)", fontsize=11, y=1.02)
        fig.tight_layout()
        return self._save(fig, "coverage_heatmaps.png")

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
            ax.plot(BIN_CENTERS, mean_per_bin, marker="o", color=color,
                    label=label, linewidth=2, markersize=7)
        ax.set_xticks(BIN_CENTERS)
        ax.set_xticklabels(BIN_LABELS, fontsize=9)
        ax.set_xlabel("Distance range", fontsize=10)
        ax.set_ylabel("Mean Absolute Error (m)", fontsize=10)
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
            ax.hist(errors, bins=50, range=(0.0, 1.0), color=color,
                    alpha=0.8, edgecolor="none")
            mean_e = float(np.mean(errors))
            ax.axvline(mean_e, color="red", linestyle="--", linewidth=1.5,
                       label=f"mean={mean_e:.3f}m")
            ax.set_xlabel("Absolute error (m)", fontsize=9)
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
        cols = ["Method", "MRE* (%)", "MRE (%)", "MAE (m)", "δ1 (%)", "Coverage (%)", "FPS", "GPU %", "GT?"]
        gt_rows, other_rows = [], []
        for name, s in self._stats.items():
            is_gt = (name == self._gt)
            row = [
                s.label,
                "—" if is_gt else f"{s.mre_pen_mean * 100:.1f}",
                "—" if is_gt else f"{s.mre_mean * 100:.1f}",
                "—" if is_gt else f"{s.mae_mean:.4f}",
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
        fig, ax = plt.subplots(figsize=(13, 1.0 + 0.55 * n))
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
        ax.set_title("Depth Quality Summary", fontsize=12, pad=10, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "summary_table.png")

    def _fig_close_range_analysis(self) -> str:
        names = list(self._r.method_names)
        if not names:
            return self._empty_fig("close_range_analysis.png", "No methods")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        labels = [self._r.method_labels.get(n, n) for n in names]
        coverages = [self._stats[n].close_range_coverage if n in self._stats else 0.0
                     for n in names]
        colors = [self._r.method_colors.get(n, "#888") for n in names]
        bars = ax1.bar(labels, coverages, color=colors, alpha=0.85, edgecolor="white")
        ax1.bar_label(bars, labels=[f"{v:.1f}%" for v in coverages], padding=3, fontsize=7)
        ax1.set_ylabel(f"Coverage at < {CLOSE_RANGE_THRESHOLD_M}m (%)", fontsize=10)
        ax1.set_title(f"Close-Range Coverage (< {CLOSE_RANGE_THRESHOLD_M} m)", fontsize=11)
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
        ax2.set_ylabel(f"Coverage at < {CLOSE_RANGE_THRESHOLD_M}m (%)", fontsize=10)
        ax2.set_title("Close-Range Coverage per Frame", fontsize=11)
        ax2.legend(fontsize=6, loc="best")
        ax2.grid(alpha=0.3)
        ax2.set_ylim(-5, 115)

        fig.suptitle("Close-Range Depth Analysis", fontsize=13, fontweight="bold")
        fig.tight_layout()
        return self._save(fig, "close_range_analysis.png")

    def _fig_timing_bars(self) -> str:
        if not self._stats:
            return self._empty_fig("timing_bars.png", "No timing data")
        names = list(self._stats.keys())
        labels = [self._stats[n].label for n in names]
        fps_vals = [min(self._stats[n].fps_mean, 200.0) for n in names]
        colors = [self._stats[n].color for n in names]
        fig, ax = plt.subplots(figsize=(8, 1.0 + 0.6 * len(names)))
        bars = ax.barh(labels, fps_vals, color=colors, alpha=0.85, edgecolor="white")
        ax.bar_label(bars, labels=[f"{f:.1f}" for f in fps_vals], padding=4, fontsize=9)
        ax.axvline(30, color="gray", linestyle="--", alpha=0.5, label="30 FPS target")
        ax.set_xlabel("Throughput (FPS)", fontsize=10)
        ax.set_title("Processing Speed by Method", fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="x", alpha=0.3)
        fig.tight_layout()
        return self._save(fig, "timing_bars.png")

    # ------------------------------------------------------------------ JSON / HTML

    def _write_json(self) -> None:
        data = {
            "source": self._r.source,
            "n_frames": self._r.n_frames,
            "resolution": f"{self._r.width}x{self._r.height}",
            "ground_truth": self._r.ground_truth_name,
            "distance_bins": BIN_LABELS,
            "model_configs": self._r.method_configs,
            "methods": {
                name: {
                    "label": s.label,
                    "mae_mean": s.mae_mean,
                    "mae_std": s.mae_std,
                    "rmse_mean": s.rmse_mean,
                    "mre_mean": s.mre_mean,
                    "delta1_mean": s.delta1_mean,
                    "coverage_mean": s.coverage_mean,
                    "fps_mean": s.fps_mean if s.fps_mean < 1e6 else -1,
                    "time_ms_mean": s.time_ms_mean,
                    "close_range_coverage": s.close_range_coverage,
                    "gpu_load_mean": s.gpu_load_mean,
                    "is_ground_truth": (name == self._gt),
                }
                for name, s in self._stats.items()
            },
        }
        (self._out / "results.json").write_text(json.dumps(data, indent=2))

    def _write_html(self, fig_paths: List[str]) -> None:
        import datetime
        ts = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        method_list = ", ".join(
            self._r.method_labels.get(n, n) for n in self._r.method_names
        )
        meta_rows = (
            f"<tr><td><b>Source</b></td><td>{self._r.source}</td></tr>"
            f"<tr><td><b>Frames</b></td><td>{self._r.n_frames}</td></tr>"
            f"<tr><td><b>Resolution</b></td><td>{self._r.width}×{self._r.height}</td></tr>"
            f"<tr><td><b>Ground truth</b></td>"
            f"<td>{self._r.method_labels.get(self._gt, self._gt)}</td></tr>"
            f"<tr><td><b>Methods evaluated</b></td><td>{method_list}</td></tr>"
            f"<tr><td><b>Generated</b></td><td>{ts}</td></tr>"
        )
        # Model configuration details for NN methods
        config_html = ""
        if self._r.method_configs:
            config_html = '\n    <div class="section">\n      <h2>Model Configuration</h2>\n      <table style="border-collapse:collapse;width:100%;font-size:.9em;">'
            config_html += '\n        <tr style="background:#2c3e50;color:white;font-weight:bold;"><td style="padding:6px 12px;">Method</td><td style="padding:6px 12px;">max_disp</td><td style="padding:6px 12px;">valid_iters</td><td style="padding:6px 12px;">Engine Resolution</td><td style="padding:6px 12px;">Engine Dir</td></tr>'
            for name, cfg in self._r.method_configs.items():
                label = self._r.method_labels.get(name, name)
                md = cfg.get("max_disp", "—")
                vi = cfg.get("valid_iters", "—")
                res = cfg.get("engine_resolution", "—")
                edir = cfg.get("engine_dir", "—")
                config_html += f'\n        <tr><td style="padding:6px 12px;">{label}</td><td style="padding:6px 12px;">{md}</td><td style="padding:6px 12px;">{vi}</td><td style="padding:6px 12px;">{res}</td><td style="padding:6px 12px;font-size:.8em;">{edir}</td></tr>'
            config_html += '\n      </table>\n    </div>'
        captions = {
            "depth_comparison.png":
                "Side-by-side depth maps from a single representative frame. Invalid pixels are black.",
            "error_maps.png":
                "Per-pixel absolute error |pred − GT| clipped at 0.1 m. Brighter = more error.",
            "coverage_heatmaps.png":
                "Fraction of frames each pixel has valid depth, averaged over all benchmark frames.",
            "distance_error_curve.png":
                "Mean Absolute Error (MAE) broken down by distance range.",
            "error_histograms.png":
                "Distribution of per-pixel absolute errors from the stored visualisation frames.",
            "summary_table.png":
                "Aggregate quality metrics — see legend below the table for column explanations.",
            "close_range_analysis.png":
                f"Coverage and stability for objects closer than {CLOSE_RANGE_THRESHOLD_M} m. Highlights MinZ benefit.",
            "timing_bars.png":
                "Processing speed in FPS. Hardware baseline is fixed at ~30 FPS (camera frame rate).",
        }
        metric_legend = (
            '\n    <div class="legend">'
            "\n      <h3>How to Read the Summary Table</h3>"
            "\n      <table>"
            "\n        <tr><td><b>MRE* (%%)</b></td>"
            "\n            <td><b>Overall score (recommended).</b> Mean Relative Error with hole penalty &mdash; "
            "pixels where the method has no depth but ground truth does count as 100%% error. "
            "This is the fairest single metric because it penalises both inaccuracy and missing coverage. <b>Lower is better.</b></td></tr>"
            "\n        <tr><td><b>MRE (%%)</b></td>"
            "\n            <td>Mean Relative Error over valid pixels only (holes ignored). "
            "5%% means each measured pixel is ~5%% off on average. <b>Lower is better.</b></td></tr>"
            "\n        <tr><td><b>MAE (m)</b></td>"
            "\n            <td>Mean Absolute Error in meters, valid pixels only. <b>Lower is better.</b></td></tr>"
            '\n        <tr><td><b>&delta;1 (%%)</b></td>'
            "\n            <td>Percentage of valid pixels within 1.25&times; of ground truth depth. <b>Higher is better.</b> 100%% is perfect.</td></tr>"
            "\n        <tr><td><b>Coverage (%%)</b></td>"
            "\n            <td>Percentage of pixels that produced valid depth. <b>Higher is better.</b> "
            "MinZ improves this at close range (&lt;0.55m) by filling holes the hardware camera cannot see.</td></tr>"
            "\n        <tr><td><b>FPS</b></td>"
            "\n            <td>Processing speed (frames per second). <b>Higher is faster.</b></td></tr>"
            '\n        <tr><td><b>GT?</b></td>'
            '\n            <td>&starf; GT marks the ground truth method (NNDepth accurate). Its error columns show "&mdash;" because you don\'t compare ground truth to itself.</td></tr>'
            "\n      </table>"
            "\n    </div>"
        )
        sections = ""
        for path in fig_paths:
            caption = captions.get(path, "")
            title = path.replace(".png", "").replace("_", " ").title()
            sections += (
                f'\n    <div class="section">'
                f"\n      <h2>{title}</h2>"
                f'\n      <div class="figure-wrapper">'
                f'\n        <img src="{path}" alt="{title}">'
                f'\n        <p class="caption">{caption}</p>'
                f"\n      </div>"
                f"\n    </div>"
            )
            if path == "summary_table.png":
                sections += metric_legend
        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Depth Quality Report — rs-enhanced-depth</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: system-ui, -apple-system, sans-serif; background: #f0f2f5;
            color: #222; max-width: 1200px; margin: 0 auto; padding: 24px 16px; }}
    h1 {{ font-size: 1.8em; color: #1a2e4a; margin-bottom: 4px; }}
    .subtitle {{ color: #555; font-size: .95em; margin-bottom: 8px; }}
    h2 {{ font-size: 1.15em; color: #1a2e4a; border-bottom: 2px solid #0f3460;
          padding-bottom: 6px; margin-bottom: 16px; }}
    .meta {{ background: white; border-radius: 10px; padding: 16px 20px;
             margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,.08); }}
    .meta table {{ border-collapse: collapse; width: 100%; font-size: .9em; }}
    .meta td {{ padding: 5px 12px; }}
    .meta tr:nth-child(even) {{ background: #f5f7fa; }}
    .section {{ background: white; border-radius: 10px; padding: 20px 24px;
                margin-bottom: 24px; box-shadow: 0 2px 6px rgba(0,0,0,.08); }}
    .figure-wrapper {{ text-align: center; }}
    .figure-wrapper img {{ max-width: 100%; height: auto;
                           border: 1px solid #e0e0e0; border-radius: 6px; }}
    .caption {{ font-size: .83em; color: #666; margin-top: 10px; }}
    .legend {{ background: #e8f4fd; border: 1px solid #b3d7f0; border-radius: 10px;
               padding: 18px 22px; margin-bottom: 24px; }}
    .legend h3 {{ font-size: 1em; color: #1a5276; margin-bottom: 10px; }}
    .legend table {{ border-collapse: collapse; width: 100%; font-size: .85em; }}
    .legend td {{ padding: 5px 10px; vertical-align: top; }}
    .legend td:first-child {{ white-space: nowrap; width: 110px; }}
    .legend tr:nth-child(even) {{ background: rgba(255,255,255,.5); }}
    footer {{ text-align: center; color: #999; font-size: .8em; padding: 24px 0 8px; }}
    .pdf-btn {{
      display: inline-flex; align-items: center; gap: 7px;
      background: #0f3460; color: white; border: none; border-radius: 7px;
      padding: 9px 20px; font-size: .95em; font-weight: 600; cursor: pointer;
      margin: 12px 0 20px; text-decoration: none; transition: background .15s;
    }}
    .pdf-btn:hover {{ background: #16213e; }}
    @media print {{
      .pdf-btn {{ display: none; }}
      body {{ background: white; padding: 0; }}
      .section {{ box-shadow: none; border: 1px solid #ddd; break-inside: avoid; }}
      .meta {{ box-shadow: none; border: 1px solid #ddd; }}
    }}
  </style>
</head>
<body>
  <h1>Depth Quality Report</h1>
  <p class="subtitle">rs-enhanced-depth — multi-method depth quality analysis</p>
  <button class="pdf-btn" onclick="window.print()">&#x1F4E5; Export as PDF</button>
  <div class="meta"><table>{meta_rows}</table></div>
{config_html}
{sections}
  <footer>Generated by <code>tests/depth_report/</code> on {ts}</footer>
</body>
</html>"""
        (self._out / "index.html").write_text(html)