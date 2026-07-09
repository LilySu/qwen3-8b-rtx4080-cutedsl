# CuteDSL Strategy for Qwen3-8B on SM89

## Chip Ground Truth

The RTX 4080 Laptop is based on the **AD104 die** — same die as the desktop RTX 4070 Ti, not the desktop 4080 (AD103). Specific numbers:

| Spec | Value |
|---|---|
| Architecture | Ada Lovelace, SM89 |
| CUDA cores | 7,424 |
| Tensor cores | 232 (4th gen) |
| Memory bus | 192-bit |
| Memory bandwidth | **432 GB/s** |
| L2 cache | ~36–48 MB (AD104 cut-dependent) |
| SMEM per SM | **100 KB** (99 KB usable per block) |
| Registers per SM | 64K × 32-bit |
| Max warps per SM | 48 |
| Max thread blocks per SM | 24 |
| TGP range | 60–150 W → boost 1350–2280 MHz |

The TGP range means compute throughput varies ~1.7× with power/thermal state while memory bandwidth stays fixed. **Lock performance mode and warm up before every profiling run** or v0→v3 comparisons will be noisy. The `smem_capacity` check in `flash_attention_v2.py:169` is hardcoded to `sm_80` (164 KB) and will not warn correctly — verify tile budgets manually against 100 KB.

---

## SM89 Capability Map

| Capability | SM89 Value | Notes |
|---|---|---|
| BF16/FP16 tensor cores | `mma.sync.aligned.m16n8k16` | Same as SM80 — mainloop reference is correct |
| FP32 throughput | **2× SM80** via duplicated FP32 pipelines | Must compile `-arch=sm_89`; SM80 binary leaves perf on the table |
| FP8 tensor cores | `mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32` | SM89-exclusive; requires CUDA ≥ 12.4 |
| FP8 variants | e4m3×e4m3, e4m3×e5m2, e5m2×e4m3, e5m2×e5m2 | All produce f32 accumulators |
| INT8 tensor cores | `mma.sync.m16n8k32.s32.s8.s8.s32` | **4× faster than FP16** on Ada consumer — highest throughput dense MMA available |
| FP32 accumulator rate | **Half rate on GeForce** | Consumer GPU penalty: FP32-accumulate MMA (FP16, BF16, FP8) runs at 50% of peak |
| `cp.async` | Yes, 4B/8B/16B variants | Same as SM80 |
| `ldmatrix` / `.trans` | Yes, SM75+ | Correct for SMEM→register MMA operand loads |
| `stmatrix` | **No — SM90+ only** | Perplexity notes and CSV claim SM89; this is wrong. SM89 epilogue stores use vectorized `st.global`. `permute_Cregs_b32_for_stsm` in quack is register permutation teaching material only. |
| `redux.sync` | Yes, SM80+ | Single-instruction warp reduction; useful for RMSNorm/QK-Norm |
| L2 persistence | `cudaAccessPropertyPersisting` | SM80+; newly impactful at 36–48 MB scale |

**SM89 features that do not exist:**

| Missing | Available from |
|---|---|
| TMA (Tensor Memory Accelerator) | SM90+ only |
| WGMMA (warpgroup MMA) | SM90+ only |
| Clusters / DSMEM | SM90+ only |
| mbarrier producer/consumer pipeline | SM90+ only |
| `stmatrix.sync` | SM90+ only |

---

## The Number That Reorganizes the Plan: Decode Is Bandwidth-Bound

At batch size 1, every generated token streams every weight byte from VRAM once. Roofline on 432 GB/s:

| Precision | Weight size | Theoretical max tok/s | Realistic |
|---|---|---|---|
| BF16 | 16.4 GB | doesn't fit | — |
| FP8 | ~8.2 GB | ~53 tok/s | 35–45 |
| INT4 | ~4.6 GB | ~94 tok/s | 65–80 |

Tensor cores are largely idle during single-token decode — an M=1 GEMV cannot feed them. The tensor-core work (steps 3–7 below) pays off in **prefill and speculative-decode verification**. Decode throughput is won by moving fewer bytes.

Also: KV cache at long context adds up fast. Qwen3-8B: 36 layers × 2 × 8 KV heads × head_dim 128 = **144 KB/token in BF16** → ~4.7 GB at 32K context. FP8 weights + BF16 KV at long context blows past 12 GB. Plan for FP8 KV too. The lm_head alone is `151,936 × 4096 × 2 bytes ≈ 1.2 GB BF16` — quantize it.

---

## FP32 Accumulator Half-Rate Penalty (GeForce-Specific, Critical)

On consumer Ada GPUs, FP32 accumulation for FP16/BF16/FP8 MMA runs at **half the rate of FP16 accumulation**. This is the defining constraint that separates Ada GeForce from A100/H100.

**Workaround — two-stage accumulation:**
- Use FP16/FP16 MMA instructions inside the mainloop (accumulate in FP16 registers)
- Every K-chunk (e.g., every 4–8 MMA tiles), promote partial sums to FP32 registers
- Final accumulator is FP32

This pattern achieves ~36% faster than cuBLAS FP16/FP32 on Ada at ~10× smaller error than pure FP16/FP16. Apply it at step 4 of the build order. It transfers directly to FP8 (step 9) since FP8 MMA also uses FP32 accumulators that hit the same half-rate penalty.

---

## What to Ignore from quack Entirely (SM90/SM100 Only)

| File | Why |
|---|---|
| `gemm_sm90.py`, `gemm_sm100.py`, `gemm_sm120.py` | WGMMA, TMA, warpgroup specialization |
| `tensormap_manager.py` | TMA descriptor management, SM90+ only |
| `pipeline.py` | mbarrier-based producer/consumer, SM90+ |
| `tile_scheduler.py` | Persistent WGMMA tile dispatch, SM90+ |
| `reduce.py:cluster_reduce` | Clusters don't exist below SM90 |
| `epi_ops.py`, `epi_composable.py` | Epilogue class hierarchy abstracts the register→SMEM→global store path — the most important thing to write yourself |
| `autotuner.py`, `cache/` | Infra, not learning material |

## What to Use from quack

| File | What to take |
|---|---|
| `dsl/cute_tensor_indexing.py` | Import and forget. `tensor[i, :, j]` instead of `tensor[i, None, j]`. Zero semantic change. |
| `cute_dsl_utils.py` | `torch2cute_dtype_map`, `get_device_capacity()` — lookup tables, not abstractions. Use `get_device_capacity()` at runtime to select tile strategies by L2 size. |
| `layout_utils.py` | Read and use selectively. `permute_Cregs_b32_for_ldsm` shows the MMA accumulator→ldmatrix register permutation (the `stsm` variant is SM90+ teaching material only on this hardware). `convert_layout_acc_mn` shows accumulator retiling for epilogue stores. `concat_to_interleave` is directly useful for SwiGLU gated layout. |
| `gemm_sm80.py` | Read as reference for SM80 mainloop. Do not instantiate — its epilogue routes through `GemmBase`/`ComposableEpiMixin`. |
| `rmsnorm.py` | Read for the 2-pass reduction pattern. Do not use the class. |
| `reduce.py:warp_reduce`, `reduce.py:block_reduce` | ~15 lines each. Copy inline. Replace inner warp loop with `redux.sync` once the baseline works. |

---

## What to Use from cutelearning

| File | Role for Qwen3 |
|---|---|
| `cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py` | Canonical SM89 GEMM. Mainloop, swizzle derivation, SMEM reuse, K-residue handling. |
| `cutedsl_examples/cute/ampere/kernel/attention/flash_attention_v2.py` | Working FA2 for Ampere. GQA and QK-Norm are small structural changes on top of this. |
| `cutedsl_examples/cute/ampere/kernel/cta_norm.py` | RMSNorm and QK-Norm reference. CTA-per-row with tree reduction. |
| `cutedsl_examples/cute/ampere/kernel/reduce.py` | `block_reduce`, `row_reduce` — copy from here. |
| `slides_runnable/dual_gemm_swiglu_epilogue.py` | Two parallel GEMMs fused with SwiGLU activation — directly the MLP. |
| `learning/leetgpu/ampere_gemm_resource_map.md` | Component-level reference: cp.async rules, ldmatrix operand layout, MMA shapes, pipeline stage count. |
| `learning/kernels/STATUS.md` | v0→v3 progression for softmax, attention, linear, RMSNorm, RoPE, GQA. Curriculum reference. |

### Key Technical Details: tensorop_gemm.py

**Swizzle derivation** (`_make_smem_layout_AB`, lines 750-765):
```python
swizzle_bits = min(int(math.log2(major_mode_size * dtype.width / copy_bits)), 3)
layout_atom = cute.make_composed_layout(
    cute.make_swizzle(swizzle_bits, 3, 3), 0,
    cute.make_layout((8, major_mode_size), stride=(major_mode_size, 1)),
)
```
For row-major BF16: `log2(64 × 16 / 128) = 3` → `Swizzle<3,3,3>` on an `8×64` atom. Use this formula; do not hardcode.

**SMEM reuse** (lines 393-404): `SharedStorageAB` and `SharedStorageC` share one allocation via `max(size_AB, size_C)`. For SM89 at 128×128×32 BF16, 3 stages: AB = 49 KB, C = 32 KB → 81 KB total, fits in 100 KB.

**K-residue handling** (lines 338-347): shifts pointer backward by `residual_k` so the irregular tile is first, eliminating boundary checks in the mainloop.

**Pipeline** (lines 622-678): `cp_async_wait_group(num_smem_stages - 2)` keeps exactly one inflight group. Issues gmem→smem on k_block==0, pure compute on remaining k_blocks.

**cp.async cache mode** (line 188): uses `LoadCacheMode.GLOBAL` = `cp.async.cg` (bypasses L1, uses L2). Correct for weight matrices accessed once per tile. Use `cp.async.ca` (cache-all, keeps in L1) only for activations that are reused across multiple K-tile passes.

**Rasterization** (lines 244-258): groups CTAs in `raster_factor × N` patches. With 36–48 MB L2, this has much larger impact than on A100 — tune `raster_factor` aggressively and consider `cudaAccessPropertyPersisting` for weight tiles that repeat across decode steps.

### Key Technical Details: flash_attention_v2.py

**4-thread reduction, not warp reduction** (`_threadquad_reduce`, lines 1104-1122):
```python
val = op(val, cute.arch.shuffle_sync_bfly(val, offset=2, ...))
val = op(val, cute.arch.shuffle_sync_bfly(val, offset=1, ...))
```
Only two shuffles. `m16n8k16` distributes one row of S across 4 consecutive threads. Full warp reduction would be wrong here — it would reduce across rows.

**Accumulator MN view** (`_make_acc_tensor_mn_view`, lines 1070-1102): reinterprets MMA fragment layout `(atom_V, MMA_M, MMA_N)` as `((atom_V_M, MMA_M), (atom_V_N, MMA_N))`. Required for `acc_S_mn[r, None].load()` — enables per-row softmax on accumulator registers.

**rP layout bridge** (lines 883-896): converts `acc_S` shape `(4, MMA_M, MMA_N)` to `((4,2), MMA_M, MMA_N/2)` for use as A operand in the O GEMM. Pure register view, no data movement.

**V transposition** (lines 427-433): `cute.composition` logical transpose — V stays in SMEM, layout is reinterpreted. `smem_copy_atom_V` uses `LdMatrix8x8x16bOp(transpose=True)` (`.trans` ldmatrix) to handle K-major→register remapping during the load.

**SMEM check is wrong for SM89** (line 169): hardcoded to `sm_80` (164 KB). At `m=n=128, head_dim=128`: `3×128×128×2 = 96 KB` — fits in 100 KB by only 4 KB. Use `n_block_size=64` for headroom: 65 KB total.

**GQA modification** (lines 399, 404): `gQ = mQ[batch_size, None, num_head, None]`, `gK = mK[batch_size, None, num_head, None]`. For Qwen3 GQA (32Q/8KV, ratio 4): derive `kv_head = num_head // 4` for K and V. The rest of the kernel is unchanged.

**QK-Norm insertion point**: after Q and K are loaded to SMEM (`sQ`, `sK` filled), before the S GEMM. Apply per-head RMSNorm as an elementwise pass over `sQ` and `sK` — same pattern as `cta_norm.py` scoped to the per-head SMEM tile.

---

## RoPE Vectorization (Step 1 Detail)

Qwen3-8B: `head_dim=128`, layout `[batch, seq, num_heads, head_dim]` (BNHD). The head_dim dimension is innermost — each warp handles a contiguous slice of one head with coalesced 128-bit access.

- Each token-head pair = 128 BF16 = 256 bytes = **16 × 16-byte loads**
- Load paired `(cos, sin)` tables as `float4` (128-bit) per 4 BF16 pairs simultaneously
- Apply: `x_out = x_real * cos - x_imag * sin` — benefits from SM89's duplicated FP32 pipe (compile `-arch=sm_89`)
- Write-back with `STG.128` aligned stores

---

## Decode-Specific Concerns

**GEMV path (M=1):** At batch 1, the GEMM degenerates to a GEMV. The 128×128 tile is wasteful — switch to a **split-K GEMV** using warp-parallel reduction over K, then `redux.sync` for the final reduction. This targets ~50% higher memory bandwidth utilization.

**Decode attention occupancy:** 8 KV heads on 58 SMs = ~14% occupancy with one CTA per head. Use **Flash-Decoding-style split-K**: split the KV sequence across CTAs plus a reduction pass. Additionally, pack the 4 Q heads sharing a KV head into one CTA so the QK^T becomes M=4 rather than four GEMVs — recovers tensor-core utilization at small batch.

**KV cache layout for paged attention:**
```
[num_pages, 2 (K/V), num_kv_heads, page_size, head_dim]
```
16-token pages with this layout allow fully coalesced `cp.async` loads. At 4K context with FP8: `36 layers × 2 × 8 × 4096 × 128 × 1 byte ≈ 570 MB` — comfortable.

---

## L2 Persistence Strategy

With 36–48 MB L2 (vs 6 MB on A100), L2 residency strategy materially changes:

**What to pin:**
- RMSNorm weight vectors: `4096 × 2 bytes × 72 instances (36 layers × 2) = 576 KB` — trivially fit all 36 layers permanently
- RoPE frequency table: `64 × 40960 × 4 bytes ≈ 10 MB` — fits
- Combined QKV projection tile: `4096 × 3072 × 2 bytes ≈ 24 MB` — pinnable for prefill phase

```python
# Query L2 size and set persistence limit
cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, l2_size)
# Pin a weight tensor
attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting
attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming
cudaStreamSetAttribute(stream, cudaStreamAttributeAccessPolicyWindow, &attr)
```

---

## Build Order (Revised and Extended)

| Step | Kernel | SM89 primitive | Notes |
|---|---|---|---|
| 1 | RoPE | Elementwise tiling, `LDG.128`/`STG.128`, `-arch=sm_89` FP32 dual-pipe | Write new |
| 2 | RMSNorm + QK-Norm | `cp.async`, 2-pass row reduction, warp shuffle → replace with `redux.sync` | `cta_norm.py` + `reduce.py` |
| 3 | Raw GEMM (no swizzle) | `ldmatrix`, `mma.sync.m16n8k16`, feel bank conflict pain | `tensorop_gemm.py` mainloop |
| 3b | GEMV path for decode | Split-K + `redux.sync` | High priority — decode bottleneck |
| 4 | GEMM + swizzle | `Swizzle<3,3,3>`, analytical formula from `_make_smem_layout_AB` | `tensorop_gemm.py` lines 750-765 |
| 4b | Two-stage FP32 accumulation | FP16/FP16 MMA + periodic FP32 promotion | Fixes GeForce half-rate penalty |
| 5 | GEMM + cp.async double-buffer | 3-stage pipeline, `cp_async_wait_group(stages-2)` | `tensorop_gemm.py` lines 622-678 |
| 5b | L2 persistence for weight tiles | `cudaAccessPropertyPersisting` | Medium — measurable in decode |
| 6 | SwiGLU epilogue | Accumulator → gated silu → vectorized global store | `dual_gemm_swiglu_epilogue.py` |
| 7 | Flash Attention (causal, FA2) | Outer KV loop, online softmax, two fused GEMMs | `flash_attention_v2.py` |
| 8 | GQA + QK-Norm | `kv_head = num_head // 4` for K/V; RMSNorm pass on sQ/sK before S GEMM | `flash_attention_v2.py` lines 399/404 |
| 9 | INT8 GEMM | `mma.sync.m16n8k32.s32.s8.s8.s32` | **4× FP16 throughput**, highest density MMA on SM89 |
| 10 | FP8 GEMM | `mma.sync.m16n8k32.f32.e4m3.e4m3.f32` + per-tensor software absmax scale | No hardware row-wise scale on SM89; two-stage accum required; verify CuTe FP8 atoms exist before assuming |
| 11 | SageAttention-style INT8 QK / FP16 PV | INT8 `mma.sync.m16n8k32`, fused RoPE + quantize in attention prologue | Natural step after step 8; 340+ TOPS on Ada |
| 12 | Split-KV decode attention + FP8 KV | Flash-Decoding split + GQA head packing (M=4 per CTA) + on-the-fly FP8 dequant | Resolves decode occupancy at 8 KV heads |
| 13 | W4A16 dequant-fused GEMM | Interleaved weight layout, dequant in mainloop between ldmatrix and mma | Marlin/ExLlamaV2 pattern; maximizes decode tok/s |

**Before starting:** run llama.cpp (Q4_K_M and Q8_0) and vLLM/ExLlamaV2 on the machine. Record tok/s for prefill and decode at several context lengths. That gives every step an honest external yardstick.

---

## WSL2-Specific Constraints

- **Available VRAM is not 12 GB.** Windows compositor + display driver share the pool. Budget ~1–1.5 GB for OS overhead → ~10.5 GB effective. Check `nvidia-smi` before kernel launch.
- **cudaDeviceScheduleBlockingSync**: WSL2 default `cudaDeviceScheduleSpin` causes high CPU usage. Set `cudaDeviceScheduleBlockingSync` to avoid kernel scheduling stalls from the WSL virtualization layer.
- **ncu profiling**: Requires "Allow access to GPU performance counters" enabled in the Windows NVIDIA Control Panel (per non-admin user). Run `ncu -o report` from within WSL; open the `.ncu-rep` file in the Nsight Compute UI on the Windows host. Use `--launch-skip` and `--launch-count` to isolate steady-state kernel instances.
- **Clock locking**: `nvidia-smi -lgc` is often unsupported on laptop. Compensate: fixed power mode, thermal warm-up, report medians over many iterations.
- **Target metrics**: for decode kernels, track achieved DRAM GB/s (target >85% of 432 GB/s; well-tuned decode kernels reach 91–95%). For prefill GEMMs, track % of your measured MMA peak (benchmark each MMA instruction separately once to get real ceilings — scaled 4090 numbers don't apply to your TGP-limited laptop).

---

## CUDA Graphs

Qwen3-8B is 36 transformer layers → ~300+ kernel launches per decode step. Under WSL2's virtualized submission path, launch overhead is amplified. Make **CUDA graph capturability** a design constraint from day one:
- Use static shapes and pre-allocated KV cache
- Pass `CUstream` explicitly to all CuTeDSL kernels (`cute.compile` supports this)
- Capture inside `torch.cuda.CUDAGraph` for replay

Per-layer graph replay consistently outperforms eager execution for decode.

---

## Speculative Decoding

The AD104's compute-to-bandwidth ratio (hundreds of TFLOPS vs 432 GB/s) is exactly the regime where speculative decoding wins: verifying k draft tokens reuses the same weight bytes already in flight, converting idle tensor-core capacity into tokens. A Qwen3-0.6B draft model fits in ~1.2 GB FP8 — leaves room alongside the 8B model for speculative decode once M=4–8 GEMMs (step 3b) are solid.
