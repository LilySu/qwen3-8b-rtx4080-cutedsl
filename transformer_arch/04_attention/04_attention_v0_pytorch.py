"""
Attention v0: F.scaled_dot_product_attention baseline (PyTorch FlashAttention).

Qwen3-8B attention: 32 Q heads, 8 KV heads, head_dim=128.
GQA: each KV head serves 4 Q heads.
QK-norm is excluded here; it's benchmarked separately as RMSNorm.

Benchmark shapes:
  prefill: (B=1, T=seq_len, H_q=32, H_kv=8, D=128)
  decode:  (B=1, T=1,       H_q=32, H_kv=8, D=128) with KV cache context
"""
from __future__ import annotations
import math
import torch
import torch.nn.functional as F
from transformer_arch._base import Metrics, SM89_PEAK_BW_GBS, cuda_time_us

NAME = "v0_pytorch"
KERNEL = "F.scaled_dot_product_attention with repeat_interleave GQA"
CHANGE = ""

N_REP = 4  # Q heads per KV head: 32 // 8


# ── op ────────────────────────────────────────────────────────────────────────

def attention(
    q: torch.Tensor,   # (B, T_q, H_q, D)
    k: torch.Tensor,   # (B, T_kv, H_kv, D)
    v: torch.Tensor,   # (B, T_kv, H_kv, D)
    is_causal: bool = True,
) -> torch.Tensor:
    B, T_q, H_q, D = q.shape
    _, T_kv, H_kv, _ = k.shape

    # expand KV to match Q head count
    k_exp = k.repeat_interleave(N_REP, dim=2)  # (B, T_kv, H_q, D)
    v_exp = v.repeat_interleave(N_REP, dim=2)

    # SDPA expects (B, H, T, D)
    q_t = q.transpose(1, 2)
    k_t = k_exp.transpose(1, 2)
    v_t = v_exp.transpose(1, 2)

    out = F.scaled_dot_product_attention(q_t, k_t, v_t, is_causal=is_causal)
    return out.transpose(1, 2)  # (B, T_q, H_q, D)


# ── correctness ───────────────────────────────────────────────────────────────

def verify(ref_fn, q, k, v, is_causal=True) -> None:
    expected = ref_fn(q, k, v, is_causal)
    got = attention(q, k, v, is_causal)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
    is_causal: bool = True,
    warmup: int = 50, iters: int = 200,
) -> Metrics:
    B, T_q, H_q, D = q.shape
    T_kv = k.shape[1]
    H_kv = k.shape[2]

    # FLOPs: QK^T matmul + softmax + PV matmul (approximate, ignoring softmax)
    # QK: (H_q, T_q, D) × (H_q, D, T_kv) = 2 * H_q * T_q * T_kv * D
    # PV: (H_q, T_q, T_kv) × (H_q, T_kv, D) = 2 * H_q * T_q * T_kv * D
    flops = 4 * B * H_q * T_q * T_kv * D

    # memory: Q + K_exp + V_exp (read) + O (write); K/V read T_kv × H_kv × D
    elem = q.element_size()
    bytes_io = elem * (
        B * T_q * H_q * D        # Q
        + B * T_kv * H_kv * D   # K (unexpanded)
        + B * T_kv * H_kv * D   # V (unexpanded)
        + B * T_q * H_q * D     # O
    )

    time_us = cuda_time_us(lambda: attention(q, k, v, is_causal), warmup, iters)
    bw = bytes_io / (time_us * 1e-6) / 1e9
    tflops = flops / (time_us * 1e-6) / 1e12

    return Metrics(
        kernel="attention",
        version=NAME,
        shape=(B, T_q, T_kv, H_q, H_kv, D),
        dtype=str(q.dtype).replace("torch.", ""),
        time_us=time_us,
        bandwidth_gbs=bw,
        tflops=tflops,
        pct_peak_bw=bw / SM89_PEAK_BW_GBS * 100,
        pct_peak_flops=None,  # memory-bound for T_q=1
        extra={"T_q": T_q, "T_kv": T_kv},
    )
