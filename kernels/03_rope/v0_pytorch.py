"""
RoPE v0: pure PyTorch baseline using complex multiplication.

This is the exact implementation used in model/rope.py, extracted here so
the benchmark harness can compare it against future CuteDSL versions.
"""
from __future__ import annotations
import torch
from kernels._base import Metrics, SM89_PEAK_BW_GBS, cuda_time_us

NAME = "v0_pytorch"
KERNEL = "complex64 multiply via view_as_complex"
CHANGE = ""  # baseline


# ── ops ───────────────────────────────────────────────────────────────────────

def precompute_freqs_cis(head_dim: int, max_seq: int, theta: float = 1_000_000.0,
                         device: torch.device = torch.device("cpu")) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device)
    freqs = torch.outer(t, freqs)
    return torch.polar(torch.ones_like(freqs), freqs)  # (max_seq, head_dim//2) complex64


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """x: (B, T, n_heads, head_dim) → same shape."""
    dtype = x.dtype
    xc = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    freqs = freqs_cis.view(1, xc.shape[1], 1, xc.shape[-1])
    out = torch.view_as_real(xc * freqs).flatten(-2)
    return out.to(dtype)


# ── correctness ───────────────────────────────────────────────────────────────

def verify(ref_fn, x: torch.Tensor, freqs: torch.Tensor) -> None:
    expected = ref_fn(x, freqs)
    got = apply_rotary_emb(x, freqs)
    torch.testing.assert_close(got, expected, rtol=1e-3, atol=1e-3)


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(x: torch.Tensor, freqs: torch.Tensor,
              warmup: int = 50, iters: int = 200) -> Metrics:
    B, T, H, D = x.shape
    # reads: x (BF16) + freqs (complex64 = FP32 pair) for T positions
    bytes_read = x.numel() * x.element_size() + T * (D // 2) * 8
    bytes_written = x.numel() * x.element_size()
    total_bytes = bytes_read + bytes_written

    time_us = cuda_time_us(lambda: apply_rotary_emb(x, freqs), warmup, iters)
    bw = total_bytes / (time_us * 1e-6) / 1e9

    return Metrics(
        kernel="rope",
        version=NAME,
        shape=(B, T, H, D),
        dtype=str(x.dtype).replace("torch.", ""),
        time_us=time_us,
        bandwidth_gbs=bw,
        tflops=None,
        pct_peak_bw=bw / SM89_PEAK_BW_GBS * 100,
        pct_peak_flops=None,
    )
