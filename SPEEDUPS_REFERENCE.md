# Kernel Speedup Reference: Qwen3 on Ada/Ampere Hardware

Collected from public blogs, repos, and benchmarks. Hardware focus: RTX 4090, RTX 3090,
RTX 3070 Laptop — all close in architecture or TDP class to the RTX 4080 Laptop (SM89,
~380 GB/s empirical BW, ~57.5 TFLOPS BF16 at laptop TGP). Quantization results excluded.

**Implementation column key:** Triton = Python Triton JIT | CUDA = hand-written `.cu` |
CUTLASS = CuTe/CUTLASS C++ | Config = serving-system settings only

---

## Hardware Mapping

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

## Per-Kernel Speedups

### RMSNorm

| Source | Impl | Hardware | Before | After | Speedup | Notes |
|---|---|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | 40.9 µs | 7.4 µs | **5.51×** | Variance+normalize+scale fused in SRAM; HBM trips: 4 → 1 |
| [Qwen-7B triton_kernels.py](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blame/refs%2Fpr%2F11/triton_kernels.py) | Triton | Not specified | — | — | — | Block-wise power-of-2 tile; single-pass mean²+rsqrt+scale |
| [FlashRT vl-transformer-primitives](https://huggingface.co/kernels/flashrt/vl-transformer-primitives) | CUDA | RTX/H100/L40s | — | — | — | Fused Q-RMSNorm+RoPE and K-RMSNorm+RoPE+KV-write; head_dim=128 fixed; Qwen3-specific |
| [LMSYS Qwen latency](https://www.lmsys.org/blog/2026-02-11-Qwen-latency/) | CUDA | AMD MI300X | — | — | — | AddRMSNorm fused with AllReduce; ratio of 1.67× transfers |

**What makes a fused RMSNorm fast on SM89:**
- Keep the squared-sum reduction in SRAM registers (warp shuffle, 5 rounds, zero SMEM traffic)
- Broadcast `inv_rms` via one `__syncthreads()` then write in a second pass — no second HBM read of `x`
- Vectorized `float4` loads: 8 BF16 values per load instruction at the same latency as 1

The FlashRT kernel is the only Qwen3-specific CUDA RMSNorm that fuses both QK-norm and RoPE
in a single kernel and is explicitly written for the decode hot path at `head_dim=128`.

---

### Fused Norm + Residual

| Source | Impl | Hardware | Before | After | Speedup | Notes |
|---|---|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | 40.6 µs | 9.0 µs | **4.49×** | Residual add + normalize in one pass |
| [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | Triton | B200 | — | — | — | 3.9% of total decode step; confirmed present |

Saves one full HBM read + write of the hidden tensor. For Qwen3-8B at hidden=4096 and BF16:
one unfused pass reads+writes `2 × 4096 × 2 = 16 KB` per token. Across 36 layers = 576 KB/token,
or ~1.5 µs at 380 GB/s. Small at decode; matters more at large prefill batch sizes.

---

### RoPE

| Source | Impl | Hardware | Before | After | Speedup | Notes |
|---|---|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | 367.9 µs | 37.3 µs | **9.87×** | M-RoPE (3D); baseline is HF eager multi-kernel; fused in-place |
| [LMSYS Qwen latency](https://www.lmsys.org/blog/2026-02-11-Qwen-latency/) | CUDA | AMD MI300X | 11.6 µs | 5.1 µs | **2.27×** | QKNorm+RoPE fused; ratio transfers to SM89 |
| [FlashRT vl-transformer-primitives](https://huggingface.co/kernels/flashrt/vl-transformer-primitives) | CUDA | RTX/L40s/H100 | — | — | — | RoPE fused into QK-norm+KV-write; CUDA graph friendly |
| [Qwen-7B triton_kernels.py](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blame/refs%2Fpr%2F11/triton_kernels.py) | Triton | Not specified | — | — | — | Standalone Triton RoPE; `x*cos + x_rot*sin` pattern |

The 9.87× is partly kernel fusion, partly eliminating eager-mode launch overhead and
intermediate tensor allocations. A clean fused Triton RoPE on SM89 vs a tuned PyTorch
baseline is realistically 2–4×. The Qwen3-8B only applies RoPE to `rope_head_dim=64` of
each head, so the input is `[seq, 32, 64]` (Q) and `[seq, 8, 64]` (K) — small tensors
where launch overhead is a large fraction of total time.

---

### GEMM (BF16 Matrix Multiply)

| Source | Impl | Hardware | Result | Notes |
|---|---|---|---|---|
| [matmul_optimizer](https://github.com/YupengHan/matmul_optimizer) | CUDA | RTX 3070 Laptop (SM86) | 800 ms → 24.16 ms; beats CUTLASS by **−6.77%** | Fixed BF16 shape, 311 rounds, human-in-the-loop |
| [autokernel](https://github.com/RightNow-AI/autokernel) | Triton | RTX 4090 | 80–95% of cuBLAS | Triton ceiling for general GEMM |
| [autokernel](https://github.com/RightNow-AI/autokernel) | CUDA | RTX 4090 | Matches/exceeds cuBLAS | WMMA + PTX intrinsics |
| [qwen3.cu](https://github.com/gigit0000/qwen3.cu) | CUDA | Not specified (Qwen3-0.6B) | ~35–39 TPS naive; cuBLAS ~2× faster | Single-file pure CUDA; CUTLASS version planned |
| [yangwenbo SM89 blog](https://yangwenbo.com/articles/fp8-blockwise-kernel-for-sm89.html) | Triton | L40S (SM89) | Matches custom CUDA | Triton 3.4.0 now matches hand-written CUDA on SM89 |
| [maknee CUTLASS blog](https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/) | CUTLASS | RTX 3090/H100 | 1% lift from ptxas ILP | CUTLASS naming triggers ptxas instruction reordering; inconsistent gains |

**Key insight from yangwenbo:** Triton 3.4.0 with PyTorch 2.8.0 now matches hand-tuned CUDA on SM89
(L40S), removing the previous reason to reach for raw CUDA for a first implementation. Start in
Triton, only drop to CUDA if the profiler shows a gap.

**Key insight from matmul_optimizer:** Shape-specialized kernels can beat CUTLASS. The
general-purpose CUTLASS kernel leaves slack on fixed shapes. 311 rounds of profiler-guided
iteration closed a 33× gap vs a naive start and overshot CUTLASS by 6.77%.

---

### SwiGLU / MLP Activation

| Source | Impl | Hardware | Result | Notes |
|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | HBM trips: 3 → 1 | Gate+SiLU activation fused; eliminates `gate_out` and `up_out` intermediate writes |
| [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | Triton | B200 | Confirmed fused | SiLU MLP gating + multiplication fused |

For Qwen3-8B at M=2048 prefill: gate_out and up_out are each `[2048, 12288]` BF16 = 48 MB.
Unfused: write gate (48 MB) + write up (48 MB) + read both (96 MB) = 192 MB extra HBM traffic.
Fused saves those 192 MB, or ~0.51 ms at 380 GB/s, per MLP layer per forward pass.

---

### Attention / SDPA

| Source | Impl | Hardware | Result | Notes |
|---|---|---|---|---|
| [autokernel](https://github.com/RightNow-AI/autokernel) | Triton | RTX 4090 | In 80–95% cuBLAS range | FlashAttention with causal mask |
| [docs.rbln.ai](https://docs.rbln.ai/latest/software/model_serving/vllm_support/tutorial/vllm_custom_kernel.html) | Triton | RBLN NPU | No speedup numbers | `flash_attention_naive_prefill` + `flash_attention_naive_decode` |

No standalone GQA or FlashDecoding speedup numbers found for SM89 without quantization.
The attention kernel bottleneck at decode is KV-cache reads, not the attention arithmetic.

---

## System-Level (Multi-Kernel) Speedups

| Source | Impl | Hardware | Before | After | Speedup | Notes |
|---|---|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 4090 (SM89) | 1× | **5.16×** | Triton kernels + CUDA graphs + static KV | **Most directly applicable to RTX 4080 Laptop** |
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | Triton | RTX 5090 | 1× | 4.26× real-time factor | CUDA graphs + static cache alone | 4× from graphs, 0.3× from Triton kernels |
| [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | Triton | B200 | 29.5 tok/s | 92.5 tok/s | **3.1×** | CUDA graphs + Triton, batch=1 |
| [NVIDIA dev blog](https://developer.nvidia.com/blog/integrate-and-deploy-tongyi-qwen3-models-into-production-applications-with-nvidia/) | Config | Not specified | 1× | **16.04×** | TensorRT-LLM full stack (BF16 baseline vs BF16) | Max batch=128; system-level, not a single kernel |

### CUDA Graph breakdown (from qwen3.5-triton profiler, B200, Qwen3.5-27B)

| Component | % of decode time |
|---|---|
| cuBLAS matrix ops (GEMM/GEMV) | 80.4% |
| Fused DeltaNet recurrent | 5.2% |
| Fused residual + RMSNorm | 3.9% |
| Causal conv1d | 3.5% |
| Kernel launch overhead | 2.5% |

**CUDA graphs eliminate ~640 kernel launches per token** for a 64-layer model. For Qwen3-8B
(36 layers), each decode step launches roughly 36 × 5 kernels = 180 kernel invocations.
At 5–20 µs overhead each, that's 0.9–3.6 ms wasted in launch overhead per token — larger
than the RMSNorm or RoPE kernel time at decode.

---

## Qwen3-Specific Fused Kernel Inventory

These kernels exist today and target the Qwen3 architecture specifically:

| Kernel | Fuses | Impl | Source | Head-dim assumption |
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

## SM89-Specific Notes (RTX 4080 Laptop / RTX 4090 / L40S)

- **FP8 blockwise quantization is NOT natively supported on SM89.** Requires SM90 (Hopper)
  or later for hardware-native blockwise FP8. Source: yangwenbo SM89 blog. Implementing
  blockwise FP8 on SM89 requires custom kernels doing the scale logic manually.
- **Triton 3.4.0 matches hand-tuned CUDA on SM89 (L40S)** for GEMM workloads. Source:
  yangwenbo SM89 blog. Previous gap between Triton and raw CUDA on Ada has closed.
- **CUTLASS threadblock swizzle is not adjustable** via the `GemmUniversalAdapter` API,
  limiting L2 cache reuse optimization. Source: yangwenbo SM89 blog.
- **ptxas instruction reordering** triggered by kernel naming conventions: naming a kernel
  with CUTLASS-style names changes SASS output without changing PTX. Gains are 1% and
  inconsistent; not a reliable technique. Source: maknee blog (RTX 3090).

---

## What quack Has vs. Community Precedent

| Kernel | quack | Impl needed | Best community speedup | Source |
|---|---|---|---|---|
| RMSNorm | v0_pytorch | Triton | **5.51×** (vs eager) | qwen3-tts-triton (RTX 5090) |
| Fused Residual+Norm | none | Triton | **4.49×** | qwen3-tts-triton (RTX 5090) |
| QKNorm+RoPE fused | none | CUDA or Triton | **2.27×** ratio | LMSYS (MI300X); FlashRT has it in CUDA |
| Standalone RoPE | v0_pytorch | Triton | 2–4× expected | qwen-7B triton_kernels.py |
| SwiGLU fusion | none | Triton | HBM 3→1 trips | qwen3-tts-triton |
| GEMM (BF16) | cuBLAS via torch | Triton → CUDA | Triton: 80–95% cuBLAS; CUDA: can exceed | autokernel, matmul_optimizer |
| CUDA graphs | none | Config | **4–5×** decode (SM89 confirmed) | qwen3-tts-triton (RTX 4090) |

**Easiest wins in order:**
1. CUDA graphs + static KV cache — no kernel writing, pure serving config, 4× confirmed on SM89
2. Fused Residual+Norm — single Triton kernel, well-understood pattern, 4.49×
3. Fused RMSNorm — same pattern as above, 5.51×
4. Fused QKNorm+RoPE — FlashRT CUDA kernel exists for Qwen3 at head_dim=128; could copy directly

---

## Source Table

| Source | Type | Impl | Hardware | Model |
|---|---|---|---|---|
| [qwen3-tts-triton](https://github.com/newgrit1004/qwen3-tts-triton) | repo | Triton | RTX 5090, RTX 4090 (community) | Qwen3-TTS 1.7B |
| [FlashRT vl-transformer-primitives](https://huggingface.co/kernels/flashrt/vl-transformer-primitives) | kernel pkg | CUDA | RTX/H100/L40s/A100 | Qwen3 / Qwen3-VL |
| [Qwen-7B-Chat-Int4 triton_kernels.py](https://huggingface.co/Qwen/Qwen-7B-Chat-Int4/blame/refs%2Fpr%2F11/triton_kernels.py) | code | Triton | Not specified | Qwen-7B |
| [qwen3.cu](https://github.com/gigit0000/qwen3.cu) | repo | CUDA | Not specified | Qwen3-0.6B FP32 |
| [LMSYS Qwen latency blog](https://www.lmsys.org/blog/2026-02-11-Qwen-latency/) | blog | CUDA | AMD MI300X | Qwen3-235B |
| [matmul_optimizer blog](https://github.com/YupengHan/matmul_optimizer) | blog | CUDA | RTX 3070 Laptop (SM86) | BF16 GEMM fixed shape |
| [yangwenbo SM89 blockwise blog](https://yangwenbo.com/articles/fp8-blockwise-kernel-for-sm89.html) | blog | Triton/CUDA | L40S (SM89) | GEMM / FP8 kernels |
| [maknee CUTLASS blog](https://maknee.github.io/blog/2025/Maybe-Consider-Putting-Cutlass-In-Your-CUDA-Kernels/) | blog | CUTLASS | RTX 3090, H100 | General kernels |
| [autokernel](https://github.com/RightNow-AI/autokernel) | repo | Triton + CUDA | H100, A100, RTX 4090 | GPT-2, LLaMA, BERT |
| [qwen3.5-triton](https://github.com/RightNow-AI/qwen3.5-triton) | repo | Triton | B200 | Qwen3.5-27B |
| [NVIDIA developer blog](https://developer.nvidia.com/blog/integrate-and-deploy-tongyi-qwen3-models-into-production-applications-with-nvidia/) | blog | Config | Not specified | Qwen3-4B |
| [byteshape.com](https://byteshape.com/blogs/Qwen3-30B-A3B-Instruct-2507/) | blog | Config | RTX 4080 desktop (16GB) | Qwen3-30B-A3B |
| [docs.rbln.ai](https://docs.rbln.ai/latest/software/model_serving/vllm_support/tutorial/vllm_custom_kernel.html) | docs | Triton | RBLN NPU | Qwen3-0.6B |
