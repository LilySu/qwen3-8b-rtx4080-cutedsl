"""
Benchmark runner.

Usage:
  python bench/runner.py                           # all kernels, all shapes
  python bench/runner.py --kernels gemm rope       # short names (see KERNELS below)
  python bench/runner.py --regime decode           # decode or prefill

Kernel short names map to directories in order of first appearance in the
Qwen3-8B forward pass:
  rmsnorm   → kernels/01_rmsnorm/   (ln1, qk-norm, ln2, final norm)
  gemm      → kernels/02_gemm/      (q/k/v proj, o_proj, lm_head)
  rope      → kernels/03_rope/
  attention → kernels/04_attention/
  mlp       → kernels/05_mlp/       (gate/up/down projections fused)

Output:
  bench/results/<kernel>_<timestamp>.json     (machine-readable)
  stdout table                                (human-readable)

Each run appends to a per-kernel JSON so the full history is preserved.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

# make repo root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from kernels._base import SM89_PEAK_BW_GBS, SM89_PEAK_BF16_TFLOPS

RESULTS_DIR = Path(__file__).parent / "results"
DEVICE = torch.device("cuda")
DTYPE = torch.bfloat16


# ── Qwen3-8B canonical shapes ─────────────────────────────────────────────────
# (label, args_fn)  —  args_fn() returns the tensor args for benchmark()

def _decode_shapes():
    H, D = 4096, 4096
    I = 12288
    H_q, H_kv, head_dim = 32, 8, 128
    T_kv = 512  # KV-cache context length

    return {
        "rope": lambda: (
            torch.randn(1, 1, H_q, head_dim, device=DEVICE, dtype=DTYPE),
            # freqs_cis for 1 position
            torch.polar(
                torch.ones(1, head_dim // 2, device=DEVICE),
                torch.zeros(1, head_dim // 2, device=DEVICE),
            ),
        ),
        "rmsnorm": lambda: (
            torch.randn(1, D, device=DEVICE, dtype=DTYPE),
            torch.ones(D, device=DEVICE, dtype=DTYPE),
        ),
        "gemm": lambda: (
            torch.randn(1, D, device=DEVICE, dtype=DTYPE),
            torch.randn(D, D, device=DEVICE, dtype=DTYPE),
        ),
        "attention": lambda: (
            torch.randn(1, 1, H_q, head_dim, device=DEVICE, dtype=DTYPE),
            torch.randn(1, T_kv, H_kv, head_dim, device=DEVICE, dtype=DTYPE),
            torch.randn(1, T_kv, H_kv, head_dim, device=DEVICE, dtype=DTYPE),
        ),
        "mlp": lambda: (
            torch.randn(1, D, device=DEVICE, dtype=DTYPE),
            torch.randn(I, D, device=DEVICE, dtype=DTYPE),
            torch.randn(I, D, device=DEVICE, dtype=DTYPE),
            torch.randn(D, I, device=DEVICE, dtype=DTYPE),
        ),
    }


def _prefill_shapes(seq_len: int = 512):
    H, D = 4096, 4096
    I = 12288
    H_q, H_kv, head_dim = 32, 8, 128

    return {
        "rope": lambda: (
            torch.randn(1, seq_len, H_q, head_dim, device=DEVICE, dtype=DTYPE),
            torch.polar(
                torch.ones(seq_len, head_dim // 2, device=DEVICE),
                torch.zeros(seq_len, head_dim // 2, device=DEVICE),
            ),
        ),
        "rmsnorm": lambda: (
            torch.randn(seq_len, D, device=DEVICE, dtype=DTYPE),
            torch.ones(D, device=DEVICE, dtype=DTYPE),
        ),
        "gemm": lambda: (
            torch.randn(seq_len, D, device=DEVICE, dtype=DTYPE),
            torch.randn(D, D, device=DEVICE, dtype=DTYPE),
        ),
        "attention": lambda: (
            torch.randn(1, seq_len, H_q, head_dim, device=DEVICE, dtype=DTYPE),
            torch.randn(1, seq_len, H_kv, head_dim, device=DEVICE, dtype=DTYPE),
            torch.randn(1, seq_len, H_kv, head_dim, device=DEVICE, dtype=DTYPE),
        ),
        "mlp": lambda: (
            torch.randn(seq_len, D, device=DEVICE, dtype=DTYPE),
            torch.randn(I, D, device=DEVICE, dtype=DTYPE),
            torch.randn(I, D, device=DEVICE, dtype=DTYPE),
            torch.randn(D, I, device=DEVICE, dtype=DTYPE),
        ),
    }


# ── kernel registry ───────────────────────────────────────────────────────────

# Maps short CLI name → numbered directory name (inference order)
KERNELS: dict[str, str] = {
    "rmsnorm":   "01_rmsnorm",
    "gemm":      "02_gemm",
    "rope":      "03_rope",
    "attention": "04_attention",
    "mlp":       "05_mlp",
}


def _discover_versions(short_name: str):
    """Return sorted list of (version_name, module) for a kernel directory.

    Uses spec_from_file_location so the directory name doesn't need to be a
    valid Python identifier (numeric prefixes like 01_ are fine).
    """
    import importlib.util
    dir_name = KERNELS[short_name]
    kernel_dir = Path(__file__).parent.parent / "kernels" / dir_name
    versions = []
    for f in sorted(kernel_dir.glob("v*.py")):
        spec = importlib.util.spec_from_file_location(
            f"kernels__{dir_name}__{f.stem}", f
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        versions.append((f.stem, mod))
    return versions


# ── table printing ────────────────────────────────────────────────────────────

_COL = ["kernel", "version", "shape", "dtype", "time_us", "bandwidth_gbs",
        "tflops", "pct_peak_bw", "pct_peak_flops"]
_WIDTH = [10, 16, 28, 10, 10, 14, 8, 12, 14]


def _header():
    return "  ".join(c.ljust(w) for c, w in zip(_COL, _WIDTH))


def _row(m: dict) -> str:
    def fmt(k, w):
        v = m.get(k)
        if v is None:
            return "—".ljust(w)
        if isinstance(v, float):
            return f"{v:.2f}".ljust(w)
        return str(v).ljust(w)
    return "  ".join(fmt(k, w) for k, w in zip(_COL, _WIDTH))


# ── main ──────────────────────────────────────────────────────────────────────

def run(kernel_filter=None, regime="both", seq_len=512):
    all_kernels = list(KERNELS)  # ordered: rmsnorm → gemm → rope → attention → mlp
    kernels = kernel_filter if kernel_filter else all_kernels

    regimes = []
    if regime in ("decode", "both"):
        regimes.append(("decode", _decode_shapes()))
    if regime in ("prefill", "both"):
        regimes.append((f"prefill_{seq_len}", _prefill_shapes(seq_len)))

    print(_header())
    print("-" * (sum(_WIDTH) + 2 * len(_WIDTH)))

    all_results = []

    for regime_label, shapes in regimes:
        for kname in kernels:
            shape_fn = shapes.get(kname)
            if shape_fn is None:
                continue
            args = shape_fn()

            for vname, mod in _discover_versions(kname):
                result = mod.benchmark(*args)
                d = result.as_dict()
                d["regime"] = regime_label
                d["timestamp"] = datetime.utcnow().isoformat()
                all_results.append(d)
                print(_row(d))

    # save
    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out_path = RESULTS_DIR / f"run_{ts}.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--kernels", nargs="+", default=None)
    p.add_argument("--regime", choices=["decode", "prefill", "both"], default="both")
    p.add_argument("--seq-len", type=int, default=512)
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    run(args.kernels, args.regime, args.seq_len)
