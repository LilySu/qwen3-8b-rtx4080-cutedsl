"""
GEMM v0: torch.matmul baseline (routes to cuBLAS).

Qwen3-8B projection shapes:
  hidden_size = 4096, intermediate_size = 12288, head_dim = 128

  Q/K/V/O projections (attention):
    q_proj:  (4096, 4096)   — 32 heads × 128
    k_proj:  (1024, 4096)   —  8 heads × 128
    v_proj:  (1024, 4096)   —  8 heads × 128
    o_proj:  (4096, 4096)

  MLP projections:
    gate_proj: (12288, 4096)
    up_proj:   (12288, 4096)
    down_proj: (4096, 12288)

For decode (M=1) these are GEMVs; for prefill (M=seq_len) they are GEMMs.
We benchmark both regimes.
"""
from __future__ import annotations
import torch
from transformer_arch._base import Metrics, SM89_PEAK_BW_GBS, SM89_PEAK_BF16_TFLOPS, cuda_time_us

NAME = "v0_pytorch"
KERNEL = "torch.matmul → cuBLAS"
CHANGE = ""


# ── op ────────────────────────────────────────────────────────────────────────

def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """x: (M, K), weight: (N, K) → (M, N)."""
    return x @ weight.T


# ── correctness ───────────────────────────────────────────────────────────────

def verify(ref_fn, x: torch.Tensor, weight: torch.Tensor) -> None:
    expected = ref_fn(x, weight)
    got = linear(x, weight)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(x: torch.Tensor, weight: torch.Tensor,
              warmup: int = 50, iters: int = 200) -> Metrics:
    M, K = x.shape
    N = weight.shape[0]

    # memory: read A + B, write C
    bytes_io = (M * K + N * K + M * N) * x.element_size()
    flops = 2 * M * N * K  # multiply-add counted as 2

    time_us = cuda_time_us(lambda: linear(x, weight), warmup, iters)
    bw = bytes_io / (time_us * 1e-6) / 1e9
    tflops = flops / (time_us * 1e-6) / 1e12

    # classify regime: decode is bandwidth-bound, prefill is compute-bound
    arithmetic_intensity = flops / bytes_io  # FLOPs per byte
    regime = "compute" if arithmetic_intensity > SM89_PEAK_BF16_TFLOPS * 1e12 / (SM89_PEAK_BW_GBS * 1e9) else "bandwidth"

    return Metrics(
        kernel="gemm",
        version=NAME,
        shape=(M, N, K),
        dtype=str(x.dtype).replace("torch.", ""),
        time_us=time_us,
        bandwidth_gbs=bw,
        tflops=tflops,
        pct_peak_bw=bw / SM89_PEAK_BW_GBS * 100,
        pct_peak_flops=tflops / SM89_PEAK_BF16_TFLOPS * 100,
        extra={"regime": regime, "arithmetic_intensity": round(arithmetic_intensity, 1)},
    )
