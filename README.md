# qwen3-8b-rtx4080-cutedsl

From-scratch implementation of Qwen3-8B inference in PyTorch and CuteDSL, iteratively optimized for the RTX 4080 Laptop (SM89 Ada Lovelace) with per-kernel profiling at each step.

---

## What this project is

This is a ground-up implementation of the full Qwen3-8B model — no HuggingFace `transformers`, no vLLM, no CUDA C++. Every component is written first in pure PyTorch to establish a correct baseline, then replaced iteratively with hand-written CuteDSL kernels that exploit the specific hardware capabilities of the RTX 4080 Laptop (AD104 die, SM89).

The goal is not just a fast inference engine — it's to build a working understanding of each optimization by going through the bottleneck removal process manually, measuring the effect at every step, and documenting what the hardware was actually doing.

Tokenization uses [fastokens](https://github.com/nreHieW/fastokens), a Rust-backed tokenizer that loads `tokenizer.json` directly without a Python tokenizer dependency. It is compatible with Python 3.9+ via the `abi3` wheel and handles Qwen3's ChatML chat template and special tokens natively.

The CuteDSL kernel strategy is informed by [quack](https://github.com/Dao-AILab/quack) (Tri Dao's CuteDSL kernel library), analyzed carefully for what transfers to SM89 vs. what is SM90+ only. quack targets H100/B200 and uses TMA, WGMMA, clusters, and mbarrier pipelines — none of which exist on SM89. What is used from it: the `cute_tensor_indexing` helper for cleaner slice syntax, the `warp_reduce`/`block_reduce` patterns (copied inline), and `layout_utils.py` for the MMA accumulator retiling and register permutation patterns needed for the epilogue store path. The full breakdown is in [CUTEDSL_STRATEGY.md](CUTEDSL_STRATEGY.md).

**Stack:** Python · PyTorch · [NVIDIA CuteDSL](https://github.com/NVIDIA/cutlass) · [fastokens](https://github.com/nreHieW/fastokens) · safetensors

---

## Hardware target

| Spec | Value |
|------|-------|
| GPU | RTX 4080 Laptop (AD104 die) |
| Architecture | Ada Lovelace, SM89 |
| Tensor cores | 232 × 4th-gen |
| Memory bandwidth | 432 GB/s (192-bit bus) |
| VRAM | 12 GB GDDR6 (~10.5 GB effective under WSL2) |
| L2 cache | ~36–48 MB (16× larger than Ampere) |
| SMEM per SM | 100 KB (vs 164 KB on A100) |
| TGP range | 60–150 W |

Key constraint: Qwen3-8B weights are ~16 GB in BF16, which does not fit. FP8 (~8.2 GB) is the minimum viable precision for the full model.

Key opportunity: the L2 cache is large enough to keep entire KV blocks or weight slabs resident between decode steps — something that doesn't apply on Ampere.

---

## Model architecture

Qwen3-8B with the following configuration:

| Parameter | Value |
|-----------|-------|
| Layers | 36 |
| Hidden size | 4096 |
| Intermediate size | 12288 |
| Q heads | 32 |
| KV heads | 8 (GQA ratio 4) |
| Head dim | 128 |
| Vocab size | 151,936 |
| Max sequence length | 40,960 |
| RoPE theta | 1,000,000 |
| Norm | RMSNorm (layers + per-head QK-norm) |
| Activation | SwiGLU |

---

## Project structure

```
model/              pure PyTorch reference implementation (always runnable)
kernels/            CuteDSL kernel iterations, ordered by inference pass
  01_rmsnorm/       pre-attn norm, QK-norm, post-attn norm, final norm
  02_gemm/          Q/K/V projections, O projection, lm_head
  03_rope/          rotary position embeddings
  04_attention/     flash attention with GQA and QK-norm
  05_mlp/           SwiGLU (gate/up/down projections fused)
bench/
  runner.py         run all kernel versions, produce comparison table + JSON
  measure_peaks.py  empirical SM89 bandwidth and MMA throughput ceilings
  results/          timestamped JSON metric files (all committed)
profiles/           NCU binary reports (gitignored) + written findings (committed)
ITERATIONS.md       master log — one row per kernel version, links bench to profile
CUTEDSL_STRATEGY.md full SM89 hardware analysis and 13-step build order
ncu_collect.sh      NCU invocation wrapper for WSL2
```

Each kernel directory follows a version naming convention:

| Version | Meaning |
|---------|---------|
| `v0_pytorch` | PyTorch / cuBLAS baseline |
| `v1_cute_naive` | CuteDSL: tiled MMA, no pipelining |
| `v2_cute_async` | CuteDSL: `cp.async` double-buffer pipeline |
| `v3_cute_swizzle` | CuteDSL: bank-conflict-free SMEM layout |
| `v4_cute_int8` | INT8 MMA (`m16n8k32`) |
| `v5_cute_fp8` | FP8 MMA with FP16 accumulation |

---

## Tracking progress

**[ITERATIONS.md](ITERATIONS.md)** is the master optimization log. Every kernel version has a row with its timing, speedup over baseline, whether it was profiled with NCU, and a one-line finding.

The rule system:
- `bench/runner.py` runs after every new version — appends a JSON to `bench/results/`
- `ncu_collect.sh` runs only when a result needs diagnosis (below threshold or surprising)
- Every NCU profile requires a written `.md` in `profiles/` before the iteration is done
- `ITERATIONS.md` is updated by hand to connect both

---

## Project reference docs

The root-level project notes have been consolidated into a single indexable reference:
- [qwen3_sm89_cutedsl_strategy_iterations_speedups_and_research_reference.md](qwen3_sm89_cutedsl_strategy_iterations_speedups_and_research_reference.md)

## Getting started

```bash
# validate the pure PyTorch implementation (no weights needed for most checks)
python validate.py

# download Qwen3-8B weights (~16 GB, requires HuggingFace access)
python download_weights.py

# run the model
python run.py --prompt "Explain rotary position embeddings"
python run.py --chat

# measure empirical SM89 peaks (run once, update kernels/_base.py constants)
python bench/measure_peaks.py

# benchmark all kernel baselines
python bench/runner.py

# benchmark a specific kernel
python bench/runner.py --kernels rope --regime decode

# profile inference with NVIDIA GenAI-Perf / AIPerf against a local endpoint
python bench/genai_perf.py --serve-local --weights weights --client genai-perf \
  --input-tokens 256 --output-tokens 128 --num-prompts 32

# or benchmark an existing OpenAI-compatible endpoint
python bench/genai_perf.py --endpoint http://127.0.0.1:8000/v1/chat/completions \
  --client aiperf --input-tokens 256 --output-tokens 128 --num-prompts 32

# run the same benchmark inside the cutelearning:dev container
docker run --rm -v "$(pwd):/workspace" cutelearning:dev bash -lc \
  'cd /workspace && python bench/genai_perf.py --serve-local --weights weights --client genai-perf --input-tokens 256 --output-tokens 128 --num-prompts 32'

# dry-run the container benchmark command without launching inference
docker run --rm -v "$(pwd):/workspace" cutelearning:dev bash -lc \
  'cd /workspace && python bench/genai_perf.py --serve-local --weights weights --client genai-perf --input-tokens 64 --output-tokens 32 --num-prompts 2 --dry-run'
```

---

## Roadmap

The build order is defined in full detail in [CUTEDSL_STRATEGY.md](CUTEDSL_STRATEGY.md). Key milestones:

### Phase 1 — elementwise kernels (Steps 1–2)

**RoPE vectorized kernel**
`head_dim=128` is 256 bytes per token-head — exactly 16 × 128-bit loads. The PyTorch baseline uses complex64 multiplication via `view_as_complex`, which touches float32 intermediate buffers. A CuteDSL kernel loads paired `(cos, sin)` as `float4`, applies the rotation in-register, and writes back with `STG.128` aligned stores — using SM89's duplicated FP32 pipelines (compile `-arch=sm_89`).

**RMSNorm with `redux.sync`**
The PyTorch baseline computes the row mean in float32 with multiple kernel launches. A CuteDSL kernel fits one row per CTA, uses `cp.async` to stage the load, warp shuffle for the partial sum, then `redux.sync` (available SM80+) for the final single-instruction warp reduction — replacing the shuffle chain with one hardware instruction.

### Phase 2 — GEMM (Steps 3–5)

**Baseline tiled GEMM**
The first CuteDSL GEMM uses `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32` with `ldmatrix` for SMEM→register loads. Bank conflicts will be visible in the first profile — this is intentional.

**Swizzled SMEM layout**
Add `Swizzle<3,3,3>` on the `8×64` atom for row-major BF16 (derived analytically: `log2(64×16/128) = 3`). Eliminates bank conflicts on the SMEM load path.

**Two-stage FP32 accumulation (GeForce-critical)**
Consumer Ada GPUs run FP32 accumulation for BF16/FP16/FP8 MMA at **half rate**. The workaround: use FP16/FP16 accumulation inside the inner MMA loop, promote partial sums to FP32 registers every K-chunk. This achieves ~36% better throughput than cuBLAS FP16/FP32 on Ada with ~10× smaller error than pure FP16.

**`cp.async` 3-stage pipeline**
Overlap GMEM→SMEM copies with MMA computation using `cp_async_wait_group(stages-2)`. Latency of the global load is hidden behind the previous tile's compute.

**Decode GEMV path**
At batch=1, the GEMM degenerates to a GEMV — a matrix-vector product that tensor cores cannot feed efficiently. A split-K GEMV using warp-parallel reduction over K followed by `redux.sync` targets ~50% higher bandwidth utilization. This is the highest-leverage decode-path change in Phase 2.

### Phase 3 — attention (Steps 6–8)

**Flash Attention 2**
Outer loop over KV tiles, online softmax with running `(max, sum)` state in registers, fused QK and PV GEMMs. SMEM tile size must be `n_block_size=64` (not 128) on SM89 — the standard FA2 reference assumes 164 KB (A100); SM89 has 100 KB, leaving only 4 KB of headroom at 128.

**GQA + QK-Norm**
Qwen3 uses 32 Q heads and 8 KV heads (ratio 4). In the attention kernel, KV head index is `q_head // 4`. QK-norm (per-head RMSNorm on Q and K) is inserted as an elementwise pass over `sQ` and `sK` in SMEM after loading, before the S GEMM — no additional memory round-trip.

**Split-KV decode attention**
8 KV heads launched on 58 SMs gives ~14% occupancy without intervention. Flash-Decoding-style KV splitting distributes the sequence across CTAs with a final reduction pass. Additionally, packing the 4 Q heads that share a KV head into one CTA converts four GEMVs into a single M=4 matmul, recovering tensor-core utilization.

### Phase 4 — quantization (Steps 9–13)

**INT8 GEMM**
`mma.sync.m16n8k32.s32.s8.s8.s32` is the highest-throughput dense MMA available on SM89 — 4× faster than FP16 with FP32 accumulation. This is the first quantized step and directly enables W8A8 (SmoothQuant-style) for prefill.

**FP8 GEMM**
`mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` — SM89-exclusive, requires CUDA ≥ 12.4. Two-stage accumulation (FP16 inner, FP32 outer) is required here too. FP8 weights at ~8.2 GB fit in 12 GB VRAM; BF16 at 16.4 GB does not.

**SageAttention-style INT8 QK / FP16 PV**
Designed specifically for SM89 consumer GPUs: quantize Q and K to INT8 for the QK^T matmul (4× throughput gain), keep PV in FP16. SageAttention2 extends this with INT4 quantization for QK, reaching 481 TOPS on the RTX 4090. Fuses RoPE and quantization into the attention kernel prologue.

**W4A16 dequant-fused GEMM**
4-bit weights at ~4.6 GB are the highest-leverage decode optimization: theoretical ~94 tok/s vs ~53 for FP8. The Marlin/ExLlamaV2 pattern interleaves the weight layout so dequantization runs in the mainloop between `ldmatrix` and `mma`, hiding dequant latency behind memory access.

### System-level (parallel with Phase 3–4)

**CUDA graphs**
Qwen3-8B has ~300+ kernel launches per decode step. Under WSL2's virtualized submission path, launch overhead compounds. Capturing the decode step as a CUDA graph and replaying it eliminates per-launch overhead. Requires static shapes and pre-allocated KV cache — design constraints to bake in early.

**L2 persistence**
With 36–48 MB of L2 (vs 6 MB on A100), tensors that repeat across decode steps can be pinned via `cudaAccessPropertyPersisting`. RMSNorm weights for all 36 layers total only 576 KB. The RoPE frequency table is ~10 MB. Both fit permanently — eliminating repeated DRAM fetches for these tensors.

**Speculative decoding**
The RTX 4080 Laptop has an extreme compute-to-bandwidth ratio: tens of TFLOPS against 432 GB/s. During single-token decode, tensor cores are largely idle. A Qwen3-0.6B draft model or EAGLE-style head turns idle compute into draft tokens, with the target model verifying k tokens in a single forward pass that reuses the same weight bytes already streamed — multiplicative improvement on top of any kernel optimization.
