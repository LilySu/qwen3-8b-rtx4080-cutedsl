# Optimization Iterations

One row per kernel version. Columns:
- **kernel / version** — matches the short name and `vN_*` filename
- **regime** — `decode` (M=1, bandwidth-bound) or `prefill` (M=seq_len, compute-bound)
- **time_us** — median kernel time from `bench/runner.py`
- **vs v0** — speedup relative to the PyTorch baseline for this kernel+regime
- **bottleneck** — what was limiting (BW / compute / occupancy / launch overhead)
- **profiled** — whether an NCU report exists in `profiles/`
- **finding** — one sentence on what the profile revealed or what changed

When to add a row: after every `bench/runner.py` run that introduces a new version.  
When to mark `profiled=yes`: after writing `profiles/<kernel>_<version>_<regime>.md`.

---

## 01 · rmsnorm

| version | regime | time_us | vs v0 | bottleneck | profiled | finding |
|---------|--------|---------|-------|------------|----------|---------|
| v0_pytorch | decode | — | 1.0× | — | no | baseline: rsqrt in FP32, cast result |
| v0_pytorch | prefill_512 | — | 1.0× | — | no | baseline |

## 02 · gemm

| version | regime | time_us | vs v0 | bottleneck | profiled | finding |
|---------|--------|---------|-------|------------|----------|---------|
| v0_pytorch | decode | — | 1.0× | — | no | baseline: cuBLAS GEMV, bandwidth-bound |
| v0_pytorch | prefill_512 | — | 1.0× | — | no | baseline: cuBLAS GEMM |

## 03 · rope

| version | regime | time_us | vs v0 | bottleneck | profiled | finding |
|---------|--------|---------|-------|------------|----------|---------|
| v0_pytorch | decode | — | 1.0× | — | no | baseline: view_as_complex multiply |
| v0_pytorch | prefill_512 | — | 1.0× | — | no | baseline |

## 04 · attention

| version | regime | time_us | vs v0 | bottleneck | profiled | finding |
|---------|--------|---------|-------|------------|----------|---------|
| v0_pytorch | decode | — | 1.0× | — | no | baseline: SDPA + repeat_interleave GQA |
| v0_pytorch | prefill_512 | — | 1.0× | — | no | baseline |

## 05 · mlp

| version | regime | time_us | vs v0 | bottleneck | profiled | finding |
|---------|--------|---------|-------|------------|----------|---------|
| v0_pytorch | decode | — | 1.0× | — | no | baseline: three separate matmuls |
| v0_pytorch | prefill_512 | — | 1.0× | — | no | baseline |
