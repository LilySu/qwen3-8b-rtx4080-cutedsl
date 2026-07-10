"""
Measure empirical SM89 peaks for the RTX 4080 Laptop.

Run this ONCE after establishing a stable thermal/power state:
  nvidia-smi -pl 136   # or whatever your max TGP is
  sleep 10             # let GPU warm up
  python bench/measure_peaks.py

Writes bench/results/peaks_<timestamp>.json and prints a table.
These measured values should replace the theoretical constants in transformer_arch/_base.py.

Measures:
  1. DRAM bandwidth — copy large tensor
  2. BF16 tensor core throughput — m16n8k16 matmul at compute-bound size
  3. INT8 tensor core throughput — m16n8k32 matmul
"""
from __future__ import annotations
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformer_arch._base import cuda_time_us

RESULTS_DIR = Path(__file__).parent / "results"
DEVICE = torch.device("cuda")


def measure_bandwidth(size_gb: float = 1.0, warmup: int = 20, iters: int = 100) -> float:
    """Returns GB/s sustained DRAM bandwidth via device-to-device copy."""
    n = int(size_gb * 1e9 / 2)  # bfloat16 = 2 bytes
    src = torch.randn(n, device=DEVICE, dtype=torch.bfloat16)
    dst = torch.empty_like(src)
    time_us = cuda_time_us(lambda: dst.copy_(src), warmup, iters)
    bytes_moved = 2 * n * 2  # read + write
    return bytes_moved / (time_us * 1e-6) / 1e9


def measure_bf16_tflops(N: int = 8192, warmup: int = 20, iters: int = 100) -> float:
    """Returns BF16 TFLOPS via square matmul at compute-bound size."""
    A = torch.randn(N, N, device=DEVICE, dtype=torch.bfloat16)
    B = torch.randn(N, N, device=DEVICE, dtype=torch.bfloat16)
    flops = 2 * N ** 3
    time_us = cuda_time_us(lambda: torch.matmul(A, B), warmup, iters)
    return flops / (time_us * 1e-6) / 1e12


def measure_int8_tops(N: int = 8192, warmup: int = 20, iters: int = 100) -> float:
    """Returns INT8 TOPS via torch._int_mm (uses INT8 tensor cores on SM89)."""
    if not hasattr(torch, "_int_mm"):
        return float("nan")
    A = torch.randint(-128, 127, (N, N), device=DEVICE, dtype=torch.int8)
    B = torch.randint(-128, 127, (N, N), device=DEVICE, dtype=torch.int8)
    ops = 2 * N ** 3
    time_us = cuda_time_us(lambda: torch._int_mm(A, B), warmup, iters)
    return ops / (time_us * 1e-6) / 1e12


def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SM count: {torch.cuda.get_device_properties(0).multi_processor_count}")
    print()

    print("Measuring bandwidth (1 GB copy)...", flush=True)
    bw = measure_bandwidth()
    print(f"  DRAM bandwidth: {bw:.1f} GB/s")

    print("Measuring BF16 TFLOPS (N=8192 matmul)...", flush=True)
    bf16 = measure_bf16_tflops()
    print(f"  BF16 TFLOPS:    {bf16:.1f}")

    print("Measuring INT8 TOPS (N=8192 matmul)...", flush=True)
    int8 = measure_int8_tops()
    print(f"  INT8 TOPS:      {int8:.1f}" if int8 == int8 else "  INT8: unavailable")

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": torch.cuda.get_device_name(0),
        "dram_bandwidth_gbs": round(bw, 1),
        "bf16_tflops": round(bf16, 1),
        "int8_tops": round(int8, 1) if int8 == int8 else None,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"peaks_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")
    print()
    print("Update SM89_PEAK_BW_GBS and SM89_PEAK_BF16_TFLOPS in transformer_arch/_base.py with these values.")


if __name__ == "__main__":
    main()
