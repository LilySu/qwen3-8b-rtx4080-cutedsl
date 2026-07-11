# Qwen3 SM89 CuTeDSL Strategy, Iteration Log, Speedups, and Research Reference

This document consolidates the root-level project notes for the Qwen3-8B on RTX 4080 Laptop work into a single, indexable reference.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Hardware Target](#hardware-target)
3. [Repository Workflow](#repository-workflow)
4. [Kernel Build Order](#kernel-build-order)
5. [Iteration Log Template](#iteration-log-template)
6. [Speedup Reference](#speedup-reference)
   - [Hardware Mapping](#hardware-mapping)
   - [SM89 Ada Hardware Facts](#sm89-ada-hardware-facts)
   - [Ada vs Ampere Kernel Behavior Differences](#ada-vs-ampere-kernel-behavior-differences)
   - [Per-Kernel Speedups](#per-kernel-speedups)
   - [System-Level Speedups](#system-level-multi-kernel-speedups)
   - [Qwen3-Specific Fused Kernel Inventory](#qwen3-specific-fused-kernel-inventory)
   - [SM89-Specific Notes](#sm89-specific-notes)
7. [CuTeDSL / CUTLASS Mapping](#cutedsl--cutlass-mapping)
   - [SM89 MMA Atoms](#sm89-mma-atoms-available-in-cute)
   - [SM89 Copy Atoms](#sm89-copy-atoms-loading-data-efficiently)
   - [Triton vs CuTeDSL Side-by-Side](#triton-vs-cutedsl-side-by-side-for-each-kernel)
   - [Where CuTeDSL Beats Triton](#where-cutedsl-beats-triton)
8. [Automated Kernel Tuning](#automated-kernel-tuning)
9. [What quack Has vs. Community Precedent](#what-quack-has-vs-community-precedent)
10. [Research Notes and Key Findings](#research-notes-and-key-findings)
11. [Profiling and Benchmarking Checklist](#profiling-and-benchmarking-checklist)
12. [Suggested Next Steps](#suggested-next-steps)
13. [Source Table](#source-table)

---

## Project Overview

This repository is a ground-up implementation of Qwen3-8B inference in PyTorch and CuteDSL, optimized for the RTX 4080 Laptop (AD104, SM89).

Key goals:
- Build a correct baseline in PyTorch.
- Replace each kernel iteratively with hand-written CuteDSL or CUDA-lean kernels.
- Measure each step with the built-in bench harness and, where useful, Nsight Compute.
- Keep the design grounded in actual SM89 hardware behavior rather than generic transformer assumptions.

The project is intended to answer two questions:
1. What does a practical SM89-specific Qwen3-8B inference path look like?
2. Where are the real throughput and latency gains on a laptop RTX 4080 instead of a datacenter GPU?

---

## Hardware Target

| Item | Value |
|---|---|
| GPU | RTX 4080 Laptop (AD104 die) |
| Architecture | Ada Lovelace, SM89 |
| Tensor cores | 232 × 4th-gen |
| Memory bandwidth | 432 GB/s |
| VRAM | ~10.5–12 GB usable under WSL2 |
| L2 cache | ~36–48 MB |
| Shared memory per SM | 100 KB (99 KB usable) |
| TGP range | 60–150 W |

Important implications:
- Decode is often memory-bound at batch size 1.
- Prefill is more compute-bound and benefits from efficient GEMM and attention kernels.
- Power/thermal state matters; reproduce results under a stable power mode.
- The large L2 cache is a strategic advantage for repeated decode and KV-cache reuse.

---

## Repository Workflow

The repository uses a simple loop:

1. Implement or improve a kernel.
2. Run the benchmark harness.
3. Record the result in the iteration log.
4. Use Nsight Compute only when a result needs diagnosis.
5. Update the documentation with the practical finding.

Core commands:

```bash
python validate.py
python download_weights.py
python run.py --prompt "Explain rotary position embeddings"
python bench/runner.py
python bench/measure_peaks.py
python bench/genai_perf.py --serve-local --weights weights --client genai-perf \
  --input-tokens 256 --output-tokens 128 --num-prompts 32
```

Container-based benchmark path:

```bash
docker run --rm -v "$(pwd):/workspace" cutelearning:dev bash -lc \
  'cd /workspace && python bench/genai_perf.py --serve-local --weights weights --client genai-perf --input-tokens 256 --output-tokens 128 --num-prompts 32'
```

---

## Kernel Build Order

The build order remains the most useful mental model for the project.

| Step | Kernel | Notes |
|---|---|---|
| 1 | RoPE | Vectorized elementwise path with coalesced loads and stores |
| 2 | RMSNorm + QK-Norm | Reduce the number of passes and use warp-level reductions where helpful |
| 3 | Raw GEMM | Start with a simple MMA path and measure the true bottleneck |
| 3b | Decode GEMV path | Single-token decode should be handled specially, not as a generic GEMM |
| 4 | GEMM with swizzle | Improve shared-memory layout and reduce bank conflicts |
| 4b | Two-stage accumulation | Important on GeForce/SM89 because FP32 accumulation is penalized |
| 5 | GEMM with cp.async pipeline | Hide GMEM-to-SMEM latency behind compute |
| 5b | L2 persistence | Use L2-residency heuristics for weight slabs and repeated decode tiles |
| 6 | SwiGLU / MLP epilogue | Fuse activation behavior where it reduces traffic |
| 7 | Flash Attention | Start from the attention skeleton and then specialize it |
| 8 | GQA + QK-Norm | Match Qwen3's 32 Q heads and 8 KV heads |
| 9 | INT8 GEMM | Strong candidate when dense tensor-core throughput matters |
| 10 | FP8 GEMM | Useful for memory reduction and capacity, but needs careful scaling |
| 11 | Split-KV decode attention | Important for the actual decode hot path |
| 12 | W4A16 or other quantized decode GEMM | Highest leverage when maximizing decode tokens/sec |

Key technical priorities:
- Keep the kernel simple enough to reason about.
- Use the hardware primitives that actually exist on SM89.
- Treat decode as a different regime from prefill.
- Measure against the real laptop ceilings rather than desktop 4090 numbers.

---

## Iteration Log Template

Use this structure for each new kernel version.

| Kernel / Version | Regime | Time (us) | vs v0 | Bottleneck | Profiled | Finding |
|---|---|---:|---:|---|---|---|
| rmsnorm / v0_pytorch | decode | — | 1.0× | — | no | Baseline |
| gemm / v1_cute_naive | prefill | — | — | — | no | Initial CuteDSL version |

Suggested columns:
- Kernel and version name
- Regime: decode, prefill, or both
- Time in microseconds
- Speedup relative to v0
- Bottleneck: bandwidth, compute, occupancy, launch overhead, or cache reuse
- Whether an Nsight Compute report exists
- One sentence summary of the learning outcome

---

## Speedup Reference

Collected from public blogs, repos, and benchmarks. Hardware focus: RTX 4090, RTX 3090,
RTX 3070 Laptop — all close in architecture or TDP class to the RTX 4080 Laptop (SM89,
~380 GB/s empirical BW, ~57.5 TFLOPS BF16 at laptop TGP). Quantization results excluded.

**Implementation column key:** Triton = Python Triton JIT | CUDA = hand-written `.cu` |
CuTeDSL = CuTe Python DSL (this project) | CUTLASS = CuTe/CUTLASS C++ | Config = serving-system settings only

---

### Hardware Mapping

| Benchmarked GPU | Arch | BW | Notes |
|---|---|---|---|
| RTX 3070 Laptop | SM86 Ampere | ~256 GB/s | Closest TDP/size analog to RTX 4080 Laptop |
| RTX 3090 | SM86 Ampere | ~936 GB/s | Same arch as 3070, much more BW |
| RTX 4080 (desktop 16GB) | SM89 Ada | ~717 GB/s | Same arch as 4080 Laptop, higher TDP |
| RTX 4090 | SM89 Ada | ~1 TB/s | **Same SM89 as 4080 Laptop**, full desktop TDP |
| L40S | SM89 Ada | ~864 GB/s | Data-center SM89; kernel results transfer to 4080 |
| RTX 5090 | SM120 Blackwell | ~1.8 TB/s | One generation ahead; Triton kernels still port |
| AMD MI300X | CDNA3 | ~5.3 TB/s | Different arch; fusion speedup *ratios* transfer |

---

### SM89 Ada Hardware Facts

These are the authoritative SM89 parameters that drive tile-size and occupancy decisions
in any CuTeDSL or CUTLASS kernel targeting the RTX 4080 Laptop.
Source: [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/archive/13.3.0/ada-tuning-guide/index.html)

| Parameter | Value | Implication |
|---|---|---|
| L1 + texture cache (combined) | 128 KB per SM | All SMEM + L1 compete for this pool |
| Shared memory carveout options | 0, 8, 16, 32, 64, 100 KB | CUDA reserves 1 KB/block; max usable = **99 KB** |
| Max resident warps per SM | 48 | At 128 threads/block → max 12 blocks resident |
| Max thread blocks per SM | 24 | |
| 32-bit registers per SM | 65,536 | 255 registers max per thread |
| L2 cache size | 64 MB (RTX 4090 desktop) / 32–48 MB (RTX 4080 Laptop) | Qwen3-8B weights >> L2; cold reads dominate |
| FP32 ops/cycle vs SM80 | **2×** | SM89 doubled FP32 throughput; compile for `-arch=sm_89` |
| BF16 tensor core instruction | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 4,096 FLOPs/warp/cycle |
| FP8 tensor core instruction | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | 8,192 FLOPs/warp/cycle (k doubles) |
| SMEM load to tensor core reg | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | Loads 4×(8×8) BF16 tiles in one warp instruction |
| Async global→SMEM copy | `cp.async.cg.shared.global` | 16-byte (128-bit) granularity; hides latency |

**SMEM budget rule for CuTeDSL double-buffered GEMM:**
- 2 × (BM × BK + BN × BK) × 2 bytes ≤ 99 KB
- At BM=128, BN=128, BK=64 (BF16): 2 × (128×64 + 128×64) × 2 = **64 KB** — fits with margin
- At BK=128: 128 KB needed — exceeds limit; reduce BM or BN to 64

---

### Ada vs Ampere Kernel Behavior Differences

Source: [triton-lang/triton issue #4906](https://github.com/triton-lang/triton/issues/4906)

| Operation | Ada RTX 4090 | Ampere A100 | Notes |
|---|---|---|---|
| INT4 GEMM speedup vs FP16 | ~3.4× | ~1.13× | Ada much better at non-standard bit widths |
| Bitshift `b >> (offs_k % 8)` | negligible overhead | dramatic slowdown | Ampere penalizes variable-operand shifts; Ada does not |
| Dynamic indexing `offs_k // N` in load | normal | high latency | Ampere memory subsystem stalls on non-uniform index patterns |

Testing a kernel on SM86 and extrapolating to the RTX 4080 Laptop may *underestimate*
the SM89 speedup, especially for kernels using bitwise ops or dynamic indexing patterns.

---

### Per-Kernel Speedups

#### RMSNorm

| Source | Impl | Hardware | Before | After | Speedup | Notes |
|---|---|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | 40.9 µs | 7.4 µs | **5.51×** | Variance+normalize+scale fused in SRAM; HBM trips: 4 → 1 |
| [Qwen-7B triton_kernels.py](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blame/refs%2Fpr%2F11/triton_kernels.py) | Triton | Not specified | — | — | — | Block-wise power-of-2 tile; single-pass mean²+rsqrt+scale |
| [FlashRT vl-transformer-primitives](https://huggingface.co/kernels/flashrt/vl-transformer-primitives) | CUDA | RTX/H100/L40s | — | — | — | Fused Q-RMSNorm+RoPE and K-RMSNorm+RoPE+KV-write; head_dim=128 fixed; Qwen3-specific |
| [LMSYS Qwen latency](https://www.lmsys.org/blog/2026-02-11-Qwen-latency/) | CUDA | AMD MI300X | — | — | — | AddRMSNorm fused with AllReduce; ratio of 1.67× transfers |

What makes a fused RMSNorm fast on SM89: keep the squared-sum reduction in registers
(warp shuffle, 5 rounds, zero SMEM traffic), broadcast `inv_rms` via one `__syncthreads()`,
write in a second pass. Vectorized `float4` loads: 8 BF16 values per instruction.

#### Fused Norm + Residual

| Source | Impl | Hardware | Before | After | Speedup | Notes |
|---|---|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | 40.6 µs | 9.0 µs | **4.49×** | Residual add + normalize in one pass |
| [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | Triton | B200 | — | — | — | 3.9% of total decode step |

Saves one full HBM read + write of the hidden tensor (16 KB/token at hidden=4096 BF16).
Across 36 layers = 576 KB/token ≈ 1.5 µs at 380 GB/s. Bigger gain at large prefill batch.

#### RoPE

| Source | Impl | Hardware | Before | After | Speedup | Notes |
|---|---|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | 367.9 µs | 37.3 µs | **9.87×** | M-RoPE (3D); HF eager baseline; fused in-place |
| [LMSYS Qwen latency](https://www.lmsys.org/blog/2026-02-11-Qwen-latency/) | CUDA | AMD MI300X | 11.6 µs | 5.1 µs | **2.27×** | QKNorm+RoPE fused; ratio transfers to SM89 |
| [FlashRT vl-transformer-primitives](https://huggingface.co/kernels/flashrt/vl-transformer-primitives) | CUDA | RTX/L40s/H100 | — | — | — | RoPE fused into QK-norm+KV-write; CUDA graph friendly |
| [Qwen-7B triton_kernels.py](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blame/refs%2Fpr%2F11/triton_kernels.py) | Triton | Not specified | — | — | — | Standalone Triton RoPE; `x*cos + x_rot*sin` pattern |

Qwen3-8B applies RoPE to `rope_head_dim=64` only: inputs `[seq, 32, 64]` (Q), `[seq, 8, 64]` (K).
These are small tensors where launch overhead dominates — CUDA graphs amplify the gain.

#### GEMM (BF16 Matrix Multiply)

| Source | Impl | Hardware | Result | Notes |
|---|---|---|---|---|
| [matmul_optimizer](https://github.com/YupengHan/matmul_optimizer) | CUDA | RTX 3070 Laptop (SM86) | 800 ms → 24.16 ms; beats CUTLASS **−6.77%** | Fixed BF16 shape, 311 rounds, human-in-the-loop |
| [autokernel](https://github.com/RightNow-AI/autokernel) | Triton | RTX 4090 | 80–95% of cuBLAS | Triton ceiling for general GEMM |
| [autokernel](https://github.com/RightNow-AI/autokernel) | CUDA | RTX 4090 | Matches/exceeds cuBLAS | WMMA + PTX intrinsics |
| [qwen3.cu](https://github.com/gigit0000/qwen3.cu) | CUDA | (Qwen3-0.6B) | ~35–39 TPS naive; cuBLAS ~2× faster | Single-file pure CUDA; CUTLASS version planned |
| [yangwenbo SM89 blog](https://yangwenbo.com/articles/fp8-blockwise-kernel-for-sm89.html) | Triton | L40S (SM89) | Matches custom CUDA | Triton 3.4.0 now matches hand-written CUDA on SM89 |
| [TritonForge (arxiv 2512.09196)](https://arxiv.org/html/2512.09196v1) | Triton | H100 | avg **1.76×** over baseline Triton | LLM-guided auto-tune of BLOCK_M/N, num_stages, num_warps |
| [maknee CUTLASS blog](https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/) | CUTLASS | RTX 3090/H100 | 1% lift from ptxas ILP | CUTLASS naming triggers ptxas instruction reordering; inconsistent |

#### SwiGLU / MLP Activation

| Source | Impl | Hardware | Result | Notes |
|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | HBM trips: 3 → 1 | Gate+SiLU fused; eliminates `gate_out` and `up_out` intermediate writes |
| [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | Triton | B200 | Confirmed fused | SiLU MLP gating + multiplication fused |

At M=2048 prefill: gate_out + up_out each = 48 MB. Unfused: 192 MB extra HBM traffic.
Fused saves ~0.51 ms per MLP layer. CuTeDSL epilogue goes further — see section 7.

#### Attention / SDPA

| Source | Impl | Hardware | Result | Notes |
|---|---|---|---|---|
| [autokernel](https://github.com/RightNow-AI/autokernel) | Triton | RTX 4090 | 80–95% cuBLAS range | FlashAttention with causal mask |
| [docs.rbln.ai](https://docs.rbln.ai/latest/software/model_serving/vllm_support/tutorial/vllm_custom_kernel.html) | Triton | RBLN NPU | No speedup numbers | `flash_attention_naive_prefill` + `flash_attention_naive_decode` |

Attention bottleneck at decode is KV-cache reads, not arithmetic. No standalone GQA
speedup numbers found for SM89 without quantization.

---

### System-Level (Multi-Kernel) Speedups

| Source | Impl | Hardware | Before | After | Speedup | Notes |
|---|---|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | **RTX 4090 (SM89)** | 1× | **5.16×** | Triton + CUDA graphs + static KV | **Most directly applicable to RTX 4080 Laptop** |
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | 1× | 4.26× | CUDA graphs + static cache alone | 4× from graphs, 0.3× from Triton kernels |
| [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | Triton | B200 | 29.5 tok/s | 92.5 tok/s | **3.1×** | CUDA graphs + Triton, batch=1 |
| [NVIDIA dev blog](https://developer.nvidia.com/blog/integrate-and-deploy-tongyi-qwen3-models-into-production-applications-with-nvidia/) | Config | Not specified | 1× | **16.04×** | TensorRT-LLM full stack (BF16 vs BF16) | Max batch=128; system-level |

**Decode step profiler breakdown (qwen3.5-triton, B200, 64-layer model):**

| Component | % of decode time |
|---|---|
| cuBLAS matrix ops (GEMM/GEMV) | 80.4% |
| Fused DeltaNet recurrent | 5.2% |
| Fused residual + RMSNorm | 3.9% |
| Kernel launch overhead | 2.5% |

CUDA graphs eliminate ~640 kernel launches per token (64-layer model). For Qwen3-8B
(36 layers) ≈ 180 launches × 5–20 µs each = 0.9–3.6 ms wasted per token in overhead alone —
larger than the entire RMSNorm or RoPE kernel time at decode.

---

### Qwen3-Specific Fused Kernel Inventory

Kernels that exist today, targeting Qwen3 architecture:

| Kernel | Fuses | Impl | Source | head_dim |
|---|---|---|---|---|
| `qwen3_q_norm_rope_qstage_bf16` | Q-RMSNorm + RoPE + Q-stage | CUDA | FlashRT | 128 (fixed) |
| `qwen3_k_norm_rope_kvwrite_bf16` | K-RMSNorm + RoPE + KV-write | CUDA | FlashRT | 128 (fixed) |
| `qwen3_k_norm_rope_kvwrite_devpos_bf16` | K-RMSNorm + RoPE + KV-write + pos tracking | CUDA | FlashRT | 128 (fixed) |
| `rms_norm_fwd_kernel` | Norm + γ-scale (single pass) | Triton | Qwen-7B triton_kernels.py | any |
| `apply_rope_fwd_kernel` | RoPE in-place | Triton | Qwen-7B triton_kernels.py | any |
| Fused RMSNorm | Variance + normalize + scale in SRAM | Triton | qwen3-tts-triton | any |
| Fused Norm+Residual | Add + normalize | Triton | qwen3-tts-triton | any |
| Fused SwiGLU | SiLU gate + up in one pass | Triton | qwen3-tts-triton | any |

---

### SM89-Specific Notes

- **FP8 blockwise quantization is NOT natively supported on SM89.** Requires SM90+. Source: yangwenbo.
- **Triton 3.4.0 matches hand-tuned CUDA on SM89** for GEMM. Source: yangwenbo SM89 blog.
- **CUTLASS `GemmUniversalAdapter` threadblock swizzle is not adjustable,** limiting L2 reuse. Source: yangwenbo.
- **ptxas instruction reordering** from CUTLASS-style kernel names: ~1% gain, inconsistent. Source: maknee.
- **Ada INT4 / bitshift performance >> Ampere.** Bitshift patterns that slow A100 to 1.13× run at 3.4× on RTX 4090. Source: triton issue #4906.
- **Compile for `-arch=sm_89`** to get 2× FP32 ops/cycle vs SM80 and FP8 tensor cores. Source: NVIDIA Ada Tuning Guide.

---

## CuTeDSL / CUTLASS Mapping

This section maps community Triton speedups to their CuTeDSL equivalents and identifies
where CuTeDSL has structural advantages over Triton for the Qwen3-8B workload.

### SM89 MMA Atoms Available in CuTe

Triton selects MMA instructions automatically. CuTeDSL lets you pick exactly.

| CuTe Atom | PTX Instruction | Shape | FLOPs/warp/cycle | Use |
|---|---|---|---|---|
| `SM89_16x8x16_F32BF16BF16F32_TN` | `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` | 16×8×16 | **4,096** | BF16 GEMM — all projection layers |
| `SM89_16x8x32_F32E4M3E4M3F32_TN` | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | 16×8×32 | **8,192** | FP8 GEMM (k-depth doubles vs BF16) |

A `tiled_mma` is a warp-level tile of these atoms:
```python
tiled_mma = cute.make_tiled_mma(
    cute.atom("SM89_16x8x16_F32BF16BF16F32_TN"),
    make_layout(make_shape(Int[2], Int[2]))   # 2×2 warp layout
)
cute.gemm(tiled_mma, rA, rB, rC)
```

### SM89 Copy Atoms: Loading Data Efficiently

| CuTe Atom | PTX | Bytes/instruction | What it replaces |
|---|---|---|---|
| `SM89_U32x4_LDSM_N` | `ldmatrix.sync.aligned.m8n8.x4.shared.b16` | 128 bytes (4 × 8×8 BF16 tiles) | 16 separate 8-byte scalar loads per warp |
| `SM80_CP_ASYNC_CACHEGLOBAL<16>` | `cp.async.cg.shared.global` | 16 bytes | Synchronous `ld.global` + SMEM store |

`ldmatrix` loads a 16×16 BF16 tile from SMEM directly into the register layout expected by
`mma.sync` — no transposing or byte-shuffling. Triton emits this behind `tl.dot`; in
CuTeDSL you wire it explicitly: `cute.copy(SM89_U32x4_LDSM_N{}, smem_tile, reg_tile)`.

### Triton vs CuTeDSL Side-by-Side for Each Kernel

#### RMSNorm (memory-bound — no tensor cores involved)

```
Triton                                CuTeDSL
──────────────────────────────────    ────────────────────────────────────────────────
@triton.jit                           @cute.kernel
def rms_norm(X, W, Y, D, eps):        def rms_norm(mX, mW, mY, D, eps):
    x  = tl.load(X + row*D + off)        # each thread owns a slice of D (D/128 elements)
    ss = tl.sum(x * x, axis=0)           tx = cute.local_partition(mX[row], threadIdx.x, 128)
    ri = tl.math.rsqrt(ss/D + eps)       x_reg = cute.copy(tx)        # 128-bit vectorized load
    tl.store(Y + ..., x * ri * w)
                                          # warp shuffle — 5 rounds, zero SMEM, zero __syncthreads
                                          ss = cute.sum(x_reg * x_reg)
                                          for delta in [16, 8, 4, 2, 1]:
                                              ss += __shfl_xor_sync(0xFFFFFFFF, ss, delta)

                                          # 4 warp totals → SMEM (4 values, not 128)
                                          if threadIdx.x % 32 == 0:
                                              smem[threadIdx.x // 32] = ss
                                          __syncthreads()              # ONE barrier

                                          if threadIdx.x == 0:
                                              smem_inv[0] = cute.math.rsqrt(
                                                  (smem[0]+smem[1]+smem[2]+smem[3]) / D + eps,
                                                  fastmath=True)
                                          __syncthreads()              # broadcast inv_rms

                                          w_reg = cute.copy(cute.local_partition(mW, threadIdx.x, 128))
                                          cute.copy(x_reg * smem_inv[0] * w_reg, tx_out)
```

CuTeDSL advantage: the 5-round warp shuffle is explicit. Triton's `tl.sum` may generate
a SMEM-based tree reduction (7 rounds + 7 `__syncthreads()` for 128 threads) instead.
The shuffle uses zero SMEM and only 2 barriers for any block size.

#### GEMM + SwiGLU Epilogue Fusion (the CuTeDSL structural advantage)

Community Triton SwiGLU fuses the activation (3 → 1 HBM trips). CuTeDSL can fuse the
activation *into the GEMM epilogue*, eliminating gate and up HBM writes entirely:

```python
# Triton: still writes `hidden` to HBM (1 write after fusing 2 reads)
gate_out = x @ W_gate.T           # GEMM → 48 MB HBM write  (M=2048)
up_out   = x @ W_up.T             # GEMM → 48 MB HBM write
hidden   = silu(gate_out) * up_out # Triton fusion: 2 reads + 1 write
out      = hidden @ W_down.T

# CuTeDSL: gate and up stay in registers during the GEMM loop
rC_gate = zeros_like_accum()      # accumulator A — register file only
rC_up   = zeros_like_accum()      # accumulator B — register file only

for k_tile in cute.range(K // BK):
    cute.copy(mX_tile, sX)        # load input once, shared by both GEMMs
    cute.copy(mWgate_tile, sWgate)
    cute.copy(mWup_tile, sWup)
    cute.gemm(tiled_mma, sX, sWgate, rC_gate)
    cute.gemm(tiled_mma, sX, sWup,   rC_up)

# Epilogue: SwiGLU in-register, one HBM write
for i in cute.range(elems_per_thread):
    g = rC_gate[i]
    hidden_reg = g / (1.0 + cute.exp(-g, fastmath=True)) * rC_up[i]
    mHidden[...] = hidden_reg     # write once; no gate_out or up_out ever touch HBM
```

**Bytes saved at M=2048 vs best community Triton result:**
- Triton fusion (3→1): still writes hidden [2048, 12288] = 48 MB
- CuTeDSL epilogue: writes nothing until down_proj — gate and up never touch HBM
- Delta: 48 MB × 2 = 96 MB saved on top of Triton fusion → ~0.25 ms at 380 GB/s per layer

This pattern is not expressible in Triton because Triton cannot hold two independent
MMA accumulators simultaneously in the same kernel block.

#### Double Buffering via `cute.make_pipeline`

```python
# Triton: num_stages=3 in @triton.autotune config (implicit)

# CuTeDSL: explicit
pipeline = cute.make_pipeline(stages=2)   # generates cp.async commit/wait groups

with pipeline.producer():
    cute.copy(AsyncCopy{}, mA_global, sA[0])   # cp.async.cg.shared.global

cute.gemm(tiled_mma, sA[1], sB[1], rC)         # compute overlaps with the async copy

with pipeline.consumer():
    pipeline.commit_wait()                       # cp.async.wait_group 0
```

The explicit pipeline makes the overlap visible and tunable — you can adjust when the
wait fires relative to the MMA issue, which Triton abstracts away.

### Where CuTeDSL Beats Triton

| Scenario | Reason |
|---|---|
| SwiGLU / dual-GEMM epilogue | Two simultaneous MMA accumulators in registers; Triton cannot express this |
| GEMM → RMSNorm epilogue | Fuse normalization in-register after GEMM; one HBM write instead of two round trips |
| Warp shuffle in reduction | Explicit 5-round shuffle; Triton may generate SMEM tree (7 rounds + 7 syncs) |
| Fixed-shape tile specialization | CuTeDSL tile sizes are compile-time; CUTLASS general kernel leaves slack |
| Precise `ldmatrix` control | Ensures tensor core register layout matches MMA operand format |
| `cp.async` pipeline timing | Explicit wait placement; Triton stages are heuristic |

### Where Triton is Better Than CuTeDSL

| Scenario | Reason |
|---|---|
| First prototype | Triton compiles in seconds; CuTeDSL requires upfront layout spec |
| Memory-bound reductions | `tl.sum` often generates good shuffle code; less to write |
| Portability | Triton auto-selects MMA atoms per architecture; CuTeDSL atoms are SM-specific |
| Triton 3.4.0 on SM89 GEMM | Matches hand-tuned CUDA; CuTeDSL advantage is now epilogue fusion, not raw speed |

---

## Automated Kernel Tuning

| Source | Impl | Method | Hardware | Result |
|---|---|---|---|---|
| [TritonForge (arxiv 2512.09196)](https://arxiv.org/html/2512.09196v1) | Triton | LLM-guided; tunes BLOCK_M/N, num_stages, num_warps, vectorization | H100 | avg **1.76×** over baseline Triton; 42.7% success rate |
| [autokernel](https://github.com/RightNow-AI/autokernel) | Triton+CUDA | Agent loop: bench → profile → modify; ~40 experiments/hour | RTX 4090, H100 | 80–95% cuBLAS (Triton); matches cuBLAS (CUDA) |
| [matmul_optimizer](https://github.com/YupengHan/matmul_optimizer) | CUDA | Human-in-the-loop; 311 rounds; profiler-guided | RTX 3070 Laptop | beat CUTLASS by **−6.77%** on fixed shape |

TritonForge parameters map directly to CuTeDSL:
- `BLOCK_M`, `BLOCK_N`, `BLOCK_K` → `BM`, `BN`, `BK` in `cute.make_tiled_mma`
- `num_stages` → `cute.make_pipeline(stages=N)`
- `num_warps` → warp layout in `make_layout(make_shape(W_M, W_N))`

---

## What quack Has vs. Community Precedent

| Kernel | quack | Impl needed | Best community speedup | Source |
|---|---|---|---|---|
| RMSNorm | v0_pytorch | CuTeDSL (explicit shuffle) or Triton | **5.51×** | qwen3-tts-triton (RTX 5090) |
| Fused Residual+Norm | none | CuTeDSL or Triton | **4.49×** | qwen3-tts-triton (RTX 5090) |
| QKNorm+RoPE fused | none | CUDA (FlashRT exists at head_dim=128) | **2.27×** ratio | LMSYS; FlashRT is copyable directly |
| SwiGLU epilogue fusion | none | **CuTeDSL** (Triton cannot hold dual accumulators) | Triton: 3→1 trips; CuTeDSL: **0 intermediate writes** | qwen3-tts-triton; CuTeDSL goes further |
| GEMM (BF16) | cuBLAS via torch | Triton first; CuTeDSL for epilogue | Triton: 80–95% cuBLAS; CUDA: can exceed | autokernel, matmul_optimizer |
| CUDA graphs | none | Config | **4–5× decode** (SM89 confirmed RTX 4090) | qwen3-tts-triton |

**Priority order:**
1. **CUDA graphs + static KV** — no kernel writing, 4× confirmed on SM89
2. **Fused Residual+Norm** — one Triton kernel, 4.49×
3. **Fused RMSNorm** — same pattern, 5.51×; CuTeDSL explicit shuffle saves 5 extra syncs
4. **Fused QKNorm+RoPE** — FlashRT CUDA kernel exists for head_dim=128; direct copy
5. **SwiGLU GEMM epilogue** — CuTeDSL dual-accumulator pattern; 96 MB saved over best Triton

---

## Source Table

| Source | Type | Impl | Hardware | Model |
|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | repo | Triton | RTX 5090, RTX 4090 | Qwen3-TTS 1.7B |
| [FlashRT vl-transformer-primitives](https://huggingface.co/kernels/flashrt/vl-transformer-primitives) | kernel pkg | CUDA | RTX/H100/L40s/A100 | Qwen3 / Qwen3-VL |
| [Qwen-7B-Chat-Int4 triton_kernels.py](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blame/refs%2Fpr%2F11/triton_kernels.py) | code | Triton | Not specified | Qwen-7B |
| [qwen3.cu](https://github.com/gigit0000/qwen3.cu) | repo | CUDA | Not specified | Qwen3-0.6B FP32 |
| [LMSYS Qwen latency blog](https://www.lmsys.org/blog/2026-02-11-Qwen-latency/) | blog | CUDA | AMD MI300X | Qwen3-235B |
| [matmul_optimizer](https://github.com/YupengHan/matmul_optimizer) | blog | CUDA | RTX 3070 Laptop (SM86) | BF16 GEMM fixed shape |
| [yangwenbo SM89 blog](https://yangwenbo.com/articles/fp8-blockwise-kernel-for-sm89.html) | blog | Triton/CUDA | L40S (SM89) | GEMM / FP8 kernels |
| [maknee CUTLASS blog](https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/) | blog | CUTLASS | RTX 3090, H100 | General kernels |
| [autokernel](https://github.com/RightNow-AI/autokernel) | repo | Triton + CUDA | H100, A100, RTX 4090 | GPT-2, LLaMA, BERT |
| [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | repo | Triton | B200 | Qwen3.5-27B |
| [NVIDIA developer blog](https://developer.nvidia.com/blog/integrate-and-deploy-tongyi-qwen3-models-into-production-applications-with-nvidia/) | blog | Config | Not specified | Qwen3-4B |
| [byteshape.com](https://byteshape.com/blogs/Qwen3-30B-A3B-Instruct-2507/) | blog | Config | RTX 4080 desktop (16GB) | Qwen3-30B-A3B |
| [TritonForge (arxiv 2512.09196)](https://arxiv.org/html/2512.09196v1) | paper | Triton | H100 | General kernels |
| [Triton bitpacked matmul issue #4906](https://github.com/triton-lang/triton/issues/4906) | issue | Triton/CUDA | RTX 4090, A100 | INT4 GEMM |
| [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/archive/13.3.0/ada-tuning-guide/index.html) | docs | — | SM89 (all Ada) | — |

---

## Research Notes and Key Findings

### SM89-specific observations
- `stmatrix` is a useful epilogue primitive on SM89, but the implementation must be chosen carefully.
- FP32 accumulation penalties matter for GeForce-class Ada hardware and change the best MMA accumulation strategy.
- The large L2 cache changes the optimization story for repeated access patterns and reusing weight slabs.

### Decode is usually bandwidth-bound
- At batch size 1, every generated token streams weight bytes from memory.
- This makes quantization and data movement strategy at least as important as raw tensor-core throughput.

### Quantization is not just a capacity trick
- FP8 and INT8 are attractive not only for fitting the model but also for improving throughput and reducing memory traffic.
- Weight-only quantization can be especially valuable for the decode path.

### Profiling advice
- Warm up the GPU and lock a stable power mode before collecting measurements.
- Measure medians rather than single runs when the GPU is thermally changing.
- Track both kernel time and achieved DRAM bandwidth where possible.

---

## Profiling and Benchmarking Checklist

Before measuring:
- Configure the laptop for a stable performance mode.
- Warm the GPU up with a few iterations.
- Run `bench/measure_peaks.py` once to get a machine-specific ceiling.

For kernel work:
- Run `python bench/runner.py` for the relevant kernel.
- Compare the result against the baseline.
- If the result is surprising, profile with Nsight Compute.

For end-to-end inference:
- Use the GenAI-Perf wrapper when you want an endpoint-style benchmark.
- Start with a moderate prompt/output size and vary it between runs.
- Capture the input/output token sizes and the model identifier along with the result.

---

## Suggested Next Steps

1. Consolidate the benchmark outputs into a single JSON/CSV report for easier comparison.
2. Add a small script to sweep prompt lengths, output lengths, and concurrency for the GenAI-Perf path.
3. Focus the next iteration on the decode path rather than the generic prefill path.
4. Align the next kernel implementation to the actual SM89 availability of the target primitives.
5. Keep this document as the single source of truth for high-level strategy and prior findings.
