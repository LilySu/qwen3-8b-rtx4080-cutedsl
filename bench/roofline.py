"""
Roofline chart generator for qwen3mma kernel iterations.

Plots measured kernel performance against hardware ceilings for the RTX 4080 Laptop
and projects each point onto H200/B200/B300 to show scaling behaviour.

Usage:
  python bench/roofline.py                                 # latest run_*.json, display
  python bench/roofline.py --run bench/results/run_X.json # specific bench file
  python bench/roofline.py --save bench/results/roofline.png
  python bench/roofline.py --regime decode                 # filter by regime
  python bench/roofline.py --hardware 4080 h200            # subset of hardware panels

Arithmetic intensity is derived from bench records as:
  intensity (FLOPs/byte) = achieved_tflops * 1e3 / achieved_bandwidth_gbs

Records where either field is None are skipped (purely memory- or compute-only tracking).
"""
from __future__ import annotations
import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np

import os
import matplotlib
if not os.environ.get("DISPLAY") and os.environ.get("MPLBACKEND") is None:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).parent / "results"


# ── hardware profiles ─────────────────────────────────────────────────────────

@dataclasses.dataclass
class HWProfile:
    name: str
    flops_fp16: float    # FLOP/s
    flops_fp8: float | None
    flops_fp4: float | None
    bw_hbm: float        # bytes/s

    @property
    def ridge_fp16(self) -> float:
        return self.flops_fp16 / self.bw_hbm

    @property
    def ridge_fp8(self) -> float | None:
        return self.flops_fp8 / self.bw_hbm if self.flops_fp8 else None

    @property
    def ridge_fp4(self) -> float | None:
        return self.flops_fp4 / self.bw_hbm if self.flops_fp4 else None


# RTX 4080 Laptop: empirical (bench/measure_peaks.py at laptop TGP)
# If you have run measure_peaks.py, update these from peaks_*.json.
RTX4080_LAPTOP = HWProfile(
    name="RTX 4080 Laptop",
    flops_fp16=57.5e12,   # measured
    flops_fp8=None,
    flops_fp4=None,
    bw_hbm=380e9,         # measured
)

H200_SXM = HWProfile(
    name="H200 SXM",
    flops_fp16=989e12,
    flops_fp8=1979e12,
    flops_fp4=None,
    bw_hbm=4.8e12,
)

B200_SXM = HWProfile(
    name="B200 SXM",
    flops_fp16=2.25e15,
    flops_fp8=4.5e15,
    flops_fp4=9.0e15,
    bw_hbm=8.0e12,
)

B300_SXM = HWProfile(
    name="B300 SXM",
    flops_fp16=3.0e15,
    flops_fp8=6.0e15,
    flops_fp4=12.0e15,
    bw_hbm=12.0e12,
)

ALL_HARDWARE: dict[str, HWProfile] = {
    "4080": RTX4080_LAPTOP,
    "h200": H200_SXM,
    "b200": B200_SXM,
    "b300": B300_SXM,
}


# ── projection ────────────────────────────────────────────────────────────────

def project_performance(
    measured_intensity: float,
    measured_flops: float,
    src: HWProfile,
    dst: HWProfile,
    precision: str = "fp16",
) -> dict:
    """Project a (intensity, achieved_flops) point from src hardware onto dst.

    Assumes the kernel maintains the same efficiency fraction of the roofline ceiling.
    Returns predicted FLOP/s and whether the operation changes memory/compute regime.
    """
    src_peak = getattr(src, f"flops_{precision}", None) or src.flops_fp16
    dst_peak = getattr(dst, f"flops_{precision}", None) or dst.flops_fp16

    src_roof = min(src_peak, src.bw_hbm * measured_intensity)
    src_efficiency = measured_flops / src_roof

    dst_roof = min(dst_peak, dst.bw_hbm * measured_intensity)
    dst_predicted = dst_roof * src_efficiency

    src_regime = "compute" if measured_intensity > src.ridge_fp16 else "memory"
    dst_ridge = dst_peak / dst.bw_hbm
    dst_regime = "compute" if measured_intensity > dst_ridge else "memory"

    return {
        "src_hw": src.name,
        "dst_hw": dst.name,
        "precision": precision,
        "intensity": measured_intensity,
        "src_achieved_tflops": measured_flops / 1e12,
        "src_efficiency_pct": src_efficiency * 100,
        "dst_predicted_tflops": dst_predicted / 1e12,
        "src_regime": src_regime,
        "dst_regime": dst_regime,
        "regime_change": src_regime != dst_regime,
    }


# ── bench → roofline points ───────────────────────────────────────────────────

def load_bench_points(run_path: Path, regime_filter: str | None = None) -> list[dict]:
    """Load a bench/results/run_*.json and convert records to roofline points.

    A record is included only when both tflops and bandwidth_gbs are non-null,
    allowing intensity to be derived as tflops * 1e3 / bandwidth_gbs.
    """
    with open(run_path) as f:
        records = json.load(f)

    points = []
    for r in records:
        if regime_filter and r.get("regime") != regime_filter:
            continue
        tflops = r.get("tflops")
        bw = r.get("bandwidth_gbs")
        if tflops is None or bw is None or bw == 0:
            continue
        intensity = tflops * 1e3 / bw  # FLOPs/byte
        achieved = tflops * 1e12        # FLOP/s
        label = f"{r['kernel']} {r['version']} ({r.get('regime', '')})"
        points.append({"label": label, "intensity": intensity, "achieved": achieved})
    return points


def _latest_run() -> Path | None:
    runs = sorted(RESULTS_DIR.glob("run_*.json"))
    return runs[-1] if runs else None


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_roofline(
    points: list[dict],
    hw_names: list[str] | None = None,
    precisions: list[str] = ("fp16",),
    save_path: str | None = None,
) -> plt.Figure:
    """Plot roofline ceilings and measured/projected kernel points.

    Each panel shows one hardware target. The leftmost panel (4080) uses measured
    points directly; remaining panels project using project_performance().
    """
    hw_keys = hw_names or list(ALL_HARDWARE.keys())
    hardware = [(k, ALL_HARDWARE[k]) for k in hw_keys if k in ALL_HARDWARE]

    fig, axes = plt.subplots(1, len(hardware), figsize=(5 * len(hardware), 5), sharey=False)
    if len(hardware) == 1:
        axes = [axes]

    x = np.logspace(-2, 4, 500)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(points), 1)))

    for ax, (hw_name, hw) in zip(axes, hardware):
        for prec in precisions:
            peak = getattr(hw, f"flops_{prec}", None)
            if peak is None:
                continue
            roof = np.minimum(peak, hw.bw_hbm * x)
            ls = "-" if prec == "fp16" else ("--" if prec == "fp8" else ":")
            ax.loglog(x, roof / 1e12, ls, linewidth=2, label=prec.upper(), color="steelblue" if prec == "fp16" else "darkorange")
            ridge = peak / hw.bw_hbm
            ax.axvline(ridge, color="gray", alpha=0.3, linestyle=":")
            ax.text(ridge * 1.05, ax.get_ylim()[0] * 1.5 if ax.get_ylim()[0] > 0 else 1e-3,
                    f"ridge\n{ridge:.0f}", fontsize=6, color="gray", va="bottom")

        for pt, color in zip(points, colors):
            if hw_name == "4080":
                achieved_tflops = pt["achieved"] / 1e12
            else:
                proj = project_performance(pt["intensity"], pt["achieved"], RTX4080_LAPTOP, hw)
                achieved_tflops = proj["dst_predicted_tflops"]
            marker = "o" if hw_name == "4080" else "^"
            alpha = 1.0 if hw_name == "4080" else 0.65
            ax.scatter(
                pt["intensity"], achieved_tflops,
                color=color, marker=marker, s=80, zorder=5, alpha=alpha,
                label=pt["label"] if ax is axes[0] else "",
            )

        ax.set_title(hw.name, fontsize=10)
        ax.set_xlabel("Arithmetic Intensity (FLOPs/byte)", fontsize=8)
        ax.set_ylabel("TFLOP/s", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, which="both", alpha=0.2)

    fig.suptitle(
        "Roofline — measured on RTX 4080 Laptop, projected to datacenter GPUs",
        fontsize=11,
    )
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved → {save_path}")
    return fig


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", type=Path, default=None,
                   help="bench/results/run_*.json to load (default: latest)")
    p.add_argument("--save", type=str, default=None,
                   help="save figure to this path (PNG/PDF)")
    p.add_argument("--regime", choices=["decode", "prefill", "both"], default=None,
                   help="filter records by regime")
    p.add_argument("--hardware", nargs="+", choices=list(ALL_HARDWARE.keys()),
                   default=None, help="hardware panels to show (default: all)")
    args = p.parse_args()

    run_path = args.run or _latest_run()
    if run_path is None:
        sys.exit("No bench/results/run_*.json found. Run bench/runner.py first.")
    print(f"Loading {run_path}")

    regime_filter = args.regime if args.regime != "both" else None
    points = load_bench_points(run_path, regime_filter)

    if not points:
        print("No roofline-plottable records found (need records with both tflops and bandwidth_gbs).")
        print("Only kernels with explicit FLOP counting appear on the roofline (e.g., gemm, mlp).")
        sys.exit(0)

    print(f"Loaded {len(points)} kernel points.")
    for pt in points:
        print(f"  {pt['label']:50s}  intensity={pt['intensity']:.2f} FLOPs/byte  "
              f"achieved={pt['achieved']/1e12:.2f} TFLOPS")

    print()
    print("Hardware ridge points:")
    for name, hw in ALL_HARDWARE.items():
        print(f"  {hw.name:20s}  FP16: {hw.ridge_fp16:.0f} FLOPs/byte")

    fig = plot_roofline(points, args.hardware, save_path=args.save)
    if not args.save:
        plt.show()


if __name__ == "__main__":
    main()
