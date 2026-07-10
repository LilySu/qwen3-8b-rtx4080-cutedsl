"""
RMSNorm v0: pure PyTorch baseline.

Qwen3 uses RMSNorm for layer norms AND for per-head QK-norm.
Shapes exercised:
  - layer norm: (B*T, hidden_size) = (seq, 4096)
  - qk norm:    (B*T*H, head_dim) = (seq*32, 128) for Q, (seq*8, 128) for K
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from transformer_arch._base import Metrics, SM89_PEAK_BW_GBS, cuda_time_us

NAME = "v0_pytorch"
KERNEL = "rsqrt + multiply in FP32, cast result"
CHANGE = ""


# ── op ────────────────────────────────────────────────────────────────────────

def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x: (..., D), weight: (D,) → same shape as x."""
    dtype = x.dtype
    x32 = x.float()
    norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)
    return (norm * weight).to(dtype)


# ── correctness ───────────────────────────────────────────────────────────────

def verify(ref_fn, x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> None:
    expected = ref_fn(x, weight, eps)
    got = rms_norm(x, weight, eps)
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6,
              warmup: int = 50, iters: int = 200) -> Metrics:
    # reads x + weight, writes output; weight is tiny relative to x
    total_bytes = 2 * x.numel() * x.element_size() + weight.numel() * weight.element_size()

    time_us = cuda_time_us(lambda: rms_norm(x, weight, eps), warmup, iters)
    bw = total_bytes / (time_us * 1e-6) / 1e9

    return Metrics(
        kernel="rmsnorm",
        version=NAME,
        shape=tuple(x.shape),
        dtype=str(x.dtype).replace("torch.", ""),
        time_us=time_us,
        bandwidth_gbs=bw,
        tflops=None,
        pct_peak_bw=bw / SM89_PEAK_BW_GBS * 100,
        pct_peak_flops=None,
    )
