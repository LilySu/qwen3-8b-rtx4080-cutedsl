"""
Measure empirical SM89 peaks for the RTX 4080 Laptop.

Run this ONCE after establishing a stable thermal/power state:
  nvidia-smi -pl 136   # or whatever your max TGP is
  sleep 10             # let GPU warm up
  python bench/measure_peaks.py

Writes bench/results/peaks_<timestamp>.json and prints a table.
Update SM89_PEAK_BW_GBS and SM89_PEAK_BF16_TFLOPS in transformer_arch/_base.py
with the measured values from this script.

Measures:
  1. DRAM bandwidth — large tensor clone (~256 MB)
  2. FP16 tensor core throughput — N=8192 matmul
  3. BF16 tensor core throughput — N=8192 matmul
  4. INT8 tensor core throughput — N=8192 via torch._int_mm
  5. GPU power draw during BF16 measurement (requires pynvml)
"""
from __future__ import annotations
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))
from transformer_arch._base import cuda_time_us, get_gpu_info

RESULTS_DIR = Path(__file__).parent / "results"
DEVICE = torch.device("cuda")

try:
    import pynvml
    pynvml.nvmlInit()
    _PYNVML = True
except Exception:
    _PYNVML = False


# ── power sampler ─────────────────────────────────────────────────────────────

def _start_power_sampler(interval: float = 0.1):
    """Returns (samples, stop_event). Set stop_event to end sampling."""
    samples: list[float] = []
    stop_event = threading.Event()
    if not _PYNVML:
        return samples, stop_event
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    def _loop():
        while not stop_event.is_set():
            try:
                samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
            except Exception:
                pass
            time.sleep(interval)

    threading.Thread(target=_loop, daemon=True).start()
    return samples, stop_event


# ── measurements ──────────────────────────────────────────────────────────────

def measure_bandwidth(warmup: int = 10, iters: int = 50) -> float:
    """Peak DRAM bandwidth in GB/s via large tensor clone (~256 MB)."""
    n = 256 * 1024 * 1024 // 4  # float32, ~256 MB
    x = torch.randn(n, device=DEVICE, dtype=torch.float32)
    time_us = cuda_time_us(lambda: x.clone(), warmup, iters)
    bytes_moved = 2 * n * 4  # read + write, float32
    return bytes_moved / (time_us * 1e-6) / 1e9


def measure_fp16_tflops(N: int = 8192, warmup: int = 10, iters: int = 50) -> float:
    """Peak FP16 TFLOPS via N×N matmul."""
    A = torch.randn(N, N, device=DEVICE, dtype=torch.float16)
    B = torch.randn(N, N, device=DEVICE, dtype=torch.float16)
    flops = 2 * N ** 3
    time_us = cuda_time_us(lambda: torch.mm(A, B), warmup, iters)
    return flops / (time_us * 1e-6) / 1e12


def measure_bf16_tflops(N: int = 8192, warmup: int = 10, iters: int = 50) -> tuple[float, list[float]]:
    """Peak BF16 TFLOPS via N×N matmul."""
    A = torch.randn(N, N, device=DEVICE, dtype=torch.bfloat16)
    B = torch.randn(N, N, device=DEVICE, dtype=torch.bfloat16)
    flops = 2 * N ** 3
    samples, stop = _start_power_sampler()
    time_us = cuda_time_us(lambda: torch.matmul(A, B), warmup, iters)
    stop.set()
    return flops / (time_us * 1e-6) / 1e12, samples


def measure_int8_tops(N: int = 8192, warmup: int = 10, iters: int = 50) -> float:
    """Peak INT8 TOPS via torch._int_mm (SM89 INT8 tensor cores)."""
    if not hasattr(torch, "_int_mm"):
        return float("nan")
    A = torch.randint(-128, 127, (N, N), device=DEVICE, dtype=torch.int8)
    B = torch.randint(-128, 127, (N, N), device=DEVICE, dtype=torch.int8)
    ops = 2 * N ** 3
    time_us = cuda_time_us(lambda: torch._int_mm(A, B), warmup, iters)
    return ops / (time_us * 1e-6) / 1e12


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    if not torch.cuda.is_available():
        sys.exit("CUDA not available")

    gpu = get_gpu_info()
    print(f"GPU:              {gpu['name']}")
    print(f"VRAM:             {gpu['vram_gb']:.1f} GB")
    print(f"SM count:         {gpu['sm_count']}")
    print(f"CUDA capability:  {gpu['cuda_capability']}")
    if not _PYNVML:
        print("(pynvml not found — power sampling disabled; pip install pynvml)")
    print()

    print("Measuring bandwidth (~256 MB clone)...", flush=True)
    bw = measure_bandwidth()
    print(f"  DRAM bandwidth: {bw:.1f} GB/s")

    print("Measuring FP16 TFLOPS (N=8192 matmul)...", flush=True)
    fp16 = measure_fp16_tflops()
    print(f"  FP16 TFLOPS:    {fp16:.1f}")

    print("Measuring BF16 TFLOPS (N=8192 matmul)...", flush=True)
    bf16, power_samples = measure_bf16_tflops()
    print(f"  BF16 TFLOPS:    {bf16:.1f}")
    if power_samples:
        avg_w = sum(power_samples) / len(power_samples)
        max_w = max(power_samples)
        print(f"  Power (avg/max): {avg_w:.0f} W / {max_w:.0f} W")

    print("Measuring INT8 TOPS (N=8192 matmul)...", flush=True)
    int8 = measure_int8_tops()
    if int8 == int8:
        print(f"  INT8 TOPS:      {int8:.1f}")
    else:
        print("  INT8: unavailable (torch._int_mm not found)")

    ridge_fp16 = fp16 * 1e12 / (bw * 1e9)
    ridge_bf16 = bf16 * 1e12 / (bw * 1e9)
    print()
    print(f"Ridge point FP16: {ridge_fp16:.0f} FLOPs/byte")
    print(f"Ridge point BF16: {ridge_bf16:.0f} FLOPs/byte")

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "gpu": gpu,
        "dram_bandwidth_gbs": round(bw, 1),
        "fp16_tflops": round(fp16, 1),
        "bf16_tflops": round(bf16, 1),
        "int8_tops": round(int8, 1) if int8 == int8 else None,
        "ridge_fp16_flops_per_byte": round(ridge_fp16, 1),
        "ridge_bf16_flops_per_byte": round(ridge_bf16, 1),
        "power_avg_w": round(sum(power_samples) / len(power_samples), 1) if power_samples else None,
        "power_max_w": round(max(power_samples), 1) if power_samples else None,
    }

    RESULTS_DIR.mkdir(exist_ok=True)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"peaks_{ts}.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved → {out}")
    print()
    print("Update these constants in transformer_arch/_base.py:")
    print(f"  SM89_PEAK_BW_GBS       = {bw:.1f}")
    print(f"  SM89_PEAK_BF16_TFLOPS  = {bf16:.1f}")


if __name__ == "__main__":
    main()
