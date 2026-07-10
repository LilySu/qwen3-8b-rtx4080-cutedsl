"""
Base class for all kernel iterations.

Convention for every v*_*.py file:
  NAME   — short label, e.g. "v0_pytorch"
  KERNEL — one-line description of what this version does differently
  CHANGE — what changed vs. the previous version (empty for v0)
  verify(ref_fn, *args) — correctness check against reference, raises on failure
  benchmark(*args, warmup=50, iters=200) -> dict — returns standard metrics dict
"""
from __future__ import annotations
import time
import torch
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Metrics:
    kernel: str
    version: str
    shape: tuple
    dtype: str
    time_us: float
    bandwidth_gbs: float | None
    tflops: float | None
    pct_peak_bw: float | None
    pct_peak_flops: float | None
    extra: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "kernel": self.kernel,
            "version": self.version,
            "shape": list(self.shape),
            "dtype": self.dtype,
            "time_us": round(self.time_us, 2),
            "bandwidth_gbs": round(self.bandwidth_gbs, 1) if self.bandwidth_gbs else None,
            "tflops": round(self.tflops, 3) if self.tflops else None,
            "pct_peak_bw": round(self.pct_peak_bw, 1) if self.pct_peak_bw else None,
            "pct_peak_flops": round(self.pct_peak_flops, 1) if self.pct_peak_flops else None,
            **self.extra,
        }


# SM89 RTX 4080 Laptop measured peaks — update with measure_peaks.py output
SM89_PEAK_BW_GBS: float = 432.0    # theoretical; measured ~390-410
SM89_PEAK_BF16_TFLOPS: float = 165.0  # measured; FP32-accum half-rate = ~82 TFLOPS effective
SM89_PEAK_INT8_TOPS: float = 330.0


def cuda_time_us(fn: Callable, warmup: int, iters: int) -> float:
    """Returns median kernel time in microseconds using CUDA events."""
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start_ev.record()
        fn()
        end_ev.record()
        torch.cuda.synchronize()
        times.append(start_ev.elapsed_time(end_ev) * 1e3)  # ms → µs

    times.sort()
    return times[len(times) // 2]  # median
