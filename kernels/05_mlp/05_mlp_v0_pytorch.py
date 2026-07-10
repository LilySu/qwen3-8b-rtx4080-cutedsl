"""
MLP v0: SwiGLU in pure PyTorch (three matmuls + silu + elementwise multiply).

Qwen3-8B: hidden=4096, intermediate=12288
  gate_proj: (12288, 4096)
  up_proj:   (12288, 4096)
  down_proj: (4096, 12288)

Total FLOPs per token: 2 × (4096×12288 + 4096×12288 + 12288×4096) = 3 × 2MN
"""
from __future__ import annotations
import torch
import torch.nn.functional as F
from kernels._base import Metrics, SM89_PEAK_BW_GBS, SM89_PEAK_BF16_TFLOPS, cuda_time_us

NAME = "v0_pytorch"
KERNEL = "silu(gate) * up → down, three separate matmuls"
CHANGE = ""


# ── op ────────────────────────────────────────────────────────────────────────

def swiglu_mlp(
    x: torch.Tensor,         # (M, hidden)
    gate_w: torch.Tensor,    # (intermediate, hidden)
    up_w: torch.Tensor,      # (intermediate, hidden)
    down_w: torch.Tensor,    # (hidden, intermediate)
) -> torch.Tensor:
    gate = F.silu(x @ gate_w.T)
    up = x @ up_w.T
    return (gate * up) @ down_w.T


# ── correctness ───────────────────────────────────────────────────────────────

def verify(ref_fn, x, gate_w, up_w, down_w) -> None:
    expected = ref_fn(x, gate_w, up_w, down_w)
    got = swiglu_mlp(x, gate_w, up_w, down_w)
    torch.testing.assert_close(got, expected, rtol=1e-2, atol=1e-2)


# ── benchmark ─────────────────────────────────────────────────────────────────

def benchmark(
    x: torch.Tensor,
    gate_w: torch.Tensor, up_w: torch.Tensor, down_w: torch.Tensor,
    warmup: int = 50, iters: int = 200,
) -> Metrics:
    M, H = x.shape
    I = gate_w.shape[0]  # intermediate

    # FLOPs: gate + up + down matmuls (silu + elementwise are negligible)
    flops = 2 * M * (H * I + H * I + I * H)

    elem = x.element_size()
    bytes_io = elem * (
        M * H          # x read (once; actually read twice but fused)
        + H * I        # gate_w
        + H * I        # up_w
        + I * H        # down_w
        + M * H        # output
    )

    time_us = cuda_time_us(lambda: swiglu_mlp(x, gate_w, up_w, down_w), warmup, iters)
    bw = bytes_io / (time_us * 1e-6) / 1e9
    tflops = flops / (time_us * 1e-6) / 1e12

    return Metrics(
        kernel="mlp",
        version=NAME,
        shape=(M, H, I),
        dtype=str(x.dtype).replace("torch.", ""),
        time_us=time_us,
        bandwidth_gbs=bw,
        tflops=tflops,
        pct_peak_bw=bw / SM89_PEAK_BW_GBS * 100,
        pct_peak_flops=tflops / SM89_PEAK_BF16_TFLOPS * 100,
    )
