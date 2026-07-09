I've pulled together the research. Below is everything organized around your actual silicon and the physics of running an 8B model on it — including several strategic points that materially change how you should read your Claude Code analysis.

## 1. Ground truth on your specific chip (not the desktop 4080)

Your GPU is not "an RTX 4080." The RTX 4080 Laptop GPU is based on the AD104 chip (same die as the desktop RTX 4070 Ti), with 7,424 CUDA cores on a 192-bit bus, 12GB GDDR6 at 432 GB/s, 58 SMs, 232 4th-gen tensor cores, and a max boost of 2280 MHz. Three consequences your analysis doesn't capture:

**Power limit is a first-order variable.** TGP is configurable between 60 and 150W, and boost ranges from 1350 MHz (at 60W) up to 2280 MHz (150W). Compute throughput can vary ~1.7x with power/thermal state while memory bandwidth stays fixed — so your compute-bound prefill numbers will be noisy while decode numbers stay stable. For profiling: pin your laptop's performance mode, plug in, and do warm-up iterations before every measurement, or your v0→v3 comparisons will be garbage.

**The giant L2 is the one Ada structural feature your doc ignores.** Each Ada SM has 128 KB of L1/SMEM and a 256 KB register file, and Ada's L2 is up to 16x larger than Ampere's — AD104 carries a 48MB-class L2 (the desktop 4070's cut-down AD104 has 36MB). On SM80, CTA rasterization for L2 reuse is a minor tweak; on SM89 with ~48MB of L2, an entire 128-wide slab of the B matrix or a whole KV block can stay L2-resident. Your `raster_factor` swizzle from `tensorop_gemm.py` is worth aggressive tuning here, and you should experiment with `cudaAccessPolicyWindow` / L2 persistence hints for the KV cache in decode. This partially substitutes for the TMA/DSMEM you don't have.

**One correction to your reference material:** `stmatrix` (`stsm`) is SM90+. quack's `permute_Cregs_b32_for_ldsm/stsm` is teaching material for the register permutation concept, but on SM89 your epilogue stores go through plain `st.shared`/vectorized global stores. `ldmatrix` (including the `.trans` variant for V) is fine — it's SM75+.

## 2. The number that should reorganize your whole plan: decode is bandwidth-bound

At batch size 1, every generated token requires streaming every weight byte from VRAM. Ceiling math on 432 GB/s:

- BF16 weights (16.4 GB): doesn't fit at all
- FP8 weights (~8.2 GB): ~50 tok/s theoretical max, realistically 35–45
- 4-bit weights (~4.6 GB): ~90 tok/s theoretical max

This means FP8 (your step 9) isn't just a capacity fix — quantization *is* the decode speed knob, and tensor cores are largely idle during decode (an M=1 GEMV can't feed them; the kernel is a memory-streaming problem). Roofline analysis of the MARLIN kernel shows LLM GEMMs hug the memory roof below batch ~64 and only reach the compute roof above it. Your tensor-core work (steps 3–7) pays off in **prefill**, prompt processing, and speculative-decode verification — decode throughput is won by moving fewer bytes.

Also note KV cache: Qwen3-8B at 36 layers × 8 KV heads × head_dim 128 costs ~144 KB/token in BF16 — ~4.7 GB at 32K context. FP8 weights + BF16 KV at long context blows past 12GB. You want FP8 (or lower) KV too.

## 3. GEMM strategies for SM89 tensor cores

**The canonical Ada worklog:** spatters.ca's "Implementing a fast Tensor Core matmul on the Ada Architecture" starts from a naive kernel and, using mma, ldmatrix, and cp.async with CUTLASS's permuted shared memory layout and an n-stage gmem→smem pipeline, finishes matching cuBLAS on a 4096³ FP16 problem. This is the exact CUDA-level twin of your CuteDSL steps 3–5 — read them side by side.

**The GeForce accumulator penalty (critical, missing from your doc):** NVIDIA confirmed in the updated whitepaper that FP32 accumulation runs at half rate on GeForce — this applies to FP16, BF16, *and* FP8 MMA. On consumer GPUs like the 4090, FP16/FP32-accumulate matmul runs at half the speed of FP16/FP16, so you choose between FP16 accumulation or 50% of peak. The workaround is two-stage accumulation: use FP16/FP16 mma instructions but accumulate results outside the mma in separate FP32 registers — this achieved 209 TFLOP/s, 36% faster than cuBLAS FP16/32, with ~10x smaller error than pure FP16/16. This same trick is what makes FP8 fast on Ada, and it's a fantastic CuteDSL exercise (a small change to your mainloop's accumulator handling).

**FP8 specifics on SM89:** The WMMA API does not expose FP8 on Ada; you must use the PTX mma.sync.aligned.m16n8k32 instruction with f8 inputs, which maps to the QMMA hardware instruction, and FP8 accumulation carries 13 fractional bits. Early cuBLASLt FP8 on the 4090 hit only half of rated throughput (~330 TFLOPS) — precisely the FP32-accumulate penalty. So your step-9 FP8 GEMM should accumulate in FP16 registers with periodic promotion to FP32 (per K-chunk), which is exactly the "two-level accumulation" pattern vLLM later adopted. Also check CUTLASS's `examples/58_ada_fp8_gemm` and the current `cute/arch/mma_sm89` atoms — FP8 CuTe atoms for Ada were historically missing, so verify what your CuteDSL version exposes before assuming you can just swap the MMA atom; you may need to define the atom against the PTX instruction yourself (another good learning exercise).

**Don't forget INT8.** On Ada consumer cards INT8 tensor cores run at full rate with INT32 accumulation — INT8 matmul on the RTX 4090/3090 is four times faster than FP16 (with FP32 accum) and two times faster than FP8. `mma.sync.m16n8k32.s32.s8.s8.s32` is the highest-throughput dense MMA you have. That opens W8A8-INT8 (SmoothQuant-style) as an alternative to FP8 for prefill.

**Measuring your actual peaks:** Lei Mao's post shows how to benchmark peak performance of each Tensor Core MMA instruction using CUTLASS/CuTe atoms — do this once on your laptop at your TGP to get real ceilings (don't trust scaled 4090 numbers), then express every kernel's performance as % of that.

## 4. Attention strategies beyond vanilla FA2

**SageAttention is the single most relevant project for you** — it was designed *for* SM89 consumer cards. It quantizes Q,K to INT8 (chosen because INT8 is 4x FP16 on the 4090/3090), keeps the PV matmul in FP16 with a low-precision FP16 accumulator to double that matmul's speed, and reaches 340 TOPS on the RTX 4090 — 52% of theoretical INT8 throughput, vs 165 TOPS peak for FlashAttention2. SageAttention2 adds per-thread INT4 quantization for QK and reaches 481 TOPS on the 4090. Their Triton kernel even fuses RoPE and quantization into the attention kernel. After your step 7/8 (FA2 + GQA), reimplementing SageAttention-v1 in CuteDSL (INT8 QK^T + FP16 PV with two-stage accumulation) is the natural step 10 — it teaches INT8 MMA, per-block quantization epilogues, and the smoothing trick, on top of the FA2 skeleton you already have.

**FP8 KV cache with correct accumulation:** vLLM found naive FP8 attention regressed a long-context needle-in-a-haystack task from 91% to 13% accuracy, fixed by a SageAttention2-style two-level accumulation writing partial results into true FP32 registers (back to 89%), at the cost of register pressure. Bake this into your kernel design from the start. FlashInfer's FP8 decode kernels show up to 2x over FP16 decode, and it also notes RoPE overhead is negligible on Ada-class GPUs due to strong CUDA-core throughput.

**Decode attention needs split-KV, not just GQA indexing.** At batch 1 with only 8 KV heads, a naive one-CTA-per-head decode kernel launches 8 blocks on a 58-SM GPU — ~14% occupancy. Flash-Decoding-style splitting of the KV sequence across CTAs plus a reduction pass is essential; llama.cpp's CUDA flash attention uses exactly this split-K parallelization across workgroups, with on-the-fly dequantization of quantized KV inside the kernel and FP16-vs-FP32 accumulator selection — its `fattn-mma` files are a good consumer-GPU reference. Also exploit GQA structurally: pack the 4 query heads sharing a KV head into one CTA so the QK^T becomes an M=4 (or M=4×spec_tokens) matrix rather than four GEMVs.

## 5. Weight quantization: the ladder below FP8

If the goal is maximum tok/s rather than just fitting, 4-bit weights beat FP8 for decode:

- **Marlin (W4A16, GPTQ):** Marlin hides dequantization overhead by overlapping data-access latency with the dequant and GEMM floating-point work, using async memory access and an optimized weight layout, achieving close to the ideal 4x speedup and stays within 5% of the 3.87x ideal up to batch 32, where other 4-bit kernels collapse after batch 4. Reading the Marlin kernel and reimplementing a simplified W4A16 GEMM in CuteDSL (interleaved weight layout, dequant in the mainloop between ldmatrix and mma) would be a superb capstone project.
- **ExLlamaV2 is specifically tuned for your arch:** on an L4 (Ada), GPTQ+ExLlamaV2 int4 hit 17.4 ms/step vs 45.2 ms for AutoAWQ+Marlin — because Marlin was tuned for SM80 while ExLlamaV2's int4 GEMMs are tuned for SM89 and recover most of the bandwidth saving the bit-width predicts. Good news for you: kernel tuning for SM89 demonstrably matters, which is your whole project thesis.
- **Even lower:** AdaLLM demonstrates NVFP4 weights on SM89 (RTX 4090) with a mandatory FP8 KV cache and a custom FP8 Triton decode kernel, noting that KV and activations often dominate VRAM at long contexts so low-bit weights alone don't materialize savings. W4A8 (QServe/QQQ) and W4A4 research kernels exist if you want to explore INT4→INT8 tensor-core paths.

A practical model recipe for your 12GB: W4 or FP8 weights, FP8 KV cache, and keep an eye on the lm_head — Qwen3's ~151k vocabulary makes the output projection ~1.2 GB in BF16 alone; quantize it too.

## 6. System-level levers that dwarf kernel micro-optimization at batch 1

**CUDA graphs.** Qwen3-8B is 36 transformer layers with 32 query heads and 8 KV heads (GQA) — that's ~300+ kernel launches per decode step, and launch overhead is amplified under WSL2's virtualized submission path. Repeated kernel launches, synchronization events, and host–device coordination accumulate across decoding steps and dominate in latency-sensitive scenarios; per-layer graph replay consistently outperforms both eager and split-graph execution for decode. CuteDSL plays well with this: you can cute.compile a kernel ahead of time, pass a CUstream explicitly, and capture it inside a torch.cuda.CUDAGraph for replay. Make graph-capturability (static shapes, pre-allocated KV) a design constraint from day one.

**Speculative decoding.** Your GPU's compute-to-bandwidth ratio is extreme (tens of TFLOPS vs 432 GB/s), which is exactly the regime where speculative decoding shines: verification of k draft tokens reuses the same weight bytes, converting idle tensor-core capacity into tokens. A Qwen3-0.6B draft model or EAGLE-style head is the highest-leverage end-to-end optimization once your kernels support M=4–8 decode GEMMs (which your GEMM work already covers).

**Baselines to beat:** run llama.cpp (Q4_K_M and Q8_0) and vLLM/ExLlamaV2 on your machine first and record tok/s for prefill and decode at several context lengths. That gives your v0→v3 progression an honest external yardstick instead of only self-relative speedups.

## 7. Profiling on Windows 11 + WSL2 specifically

- Nsight Compute (ncu) works for kernel profiling on WSL2 — measuring compute utilization and effective memory bandwidth — but requires extra setup; install the Nsight Compute UI on the Windows host and generate reports with ncu -o from within WSL. You must also enable "Allow access to GPU performance counters" for non-admin users in the Windows NVIDIA control panel, or every ncu run fails with ERR_NVGPUCTRPERM.
- Use Nsight Systems for low-overhead timeline profiling and the PyTorch profiler for richer traces with stack/shape info; use --launch-skip/--launch-count and kernel-name filters in ncu so you profile just the steady-state instances of the kernels that matter.
- On a laptop, clock locking (`nvidia-smi -lgc`) is often unsupported — compensate with fixed power mode, thermal warm-up, and reporting medians over many iterations. Track achieved DRAM GB/s for decode kernels (target: >85% of 432 — well-tuned decode attention kernels reach 91–95% bandwidth utilization) and % of your measured MMA peak for prefill GEMMs.

## 8. CuteDSL-specific resources to add to your reading list

Beyond quack and cutelearning: the official CuTe DSL examples in the CUTLASS repo (persistent dense GEMM, grouped GEMM, and fused multi-head attention) demonstrate C++-comparable performance on Ampere-class targets — the Ampere FMHA example is directly runnable on SM89. Chris Choy's "CuTe DSL Basics" covers the practical mechanics: cute.compile, dynamic vs static values, cute.printf for runtime debugging, TV layouts, and CUDA-graph integration; Simon Veitner's blog (veitner.bearblog.dev) has a series of applied CuTeDSL walkthroughs including memory-bound kernel vectorization; and Ian Barber's post documents the sharp edges — kernels are aggressively cached (rm ~/.cache/cutedsl if things look stale), multiple @cute.jit hosts in one scope can confuse MLIR, and control flow rules are strict (no return inside a kernel, initialize everything). Install via the repo's setup.sh or pip install nvidia-cutlass-dsl, matched to your CUDA toolkit version.

## Suggested amendments to your build order

Your steps 1–8 are sound. I'd revise the endgame: insert **INT8 GEMM (m16n8k32.s32.s8.s8)** before FP8, since it's the fastest MMA on your silicon and the dequant/scale plumbing transfers directly; make step 9's FP8 GEMM use **FP16 accumulators with periodic FP32 promotion** (the half-rate FP32-accum penalty is the defining GeForce constraint); then add step 10: **W4A16 dequant-fused GEMM** (Marlin/ExLlamaV2 pattern — this is what actually maximizes decode tok/s), step 11: **split-KV decode attention with FP8 KV and GQA head-packing**, and step 12: **SageAttention-style INT8 QK / FP16 PV attention** for prefill. Wrap everything in CUDA graphs and, if you're still hungry, bolt on speculative decoding.

If you want, I can turn this into a consolidated reference document, or go deeper on any single piece — e.g., the exact register-level layout for the FP8 m16n8k32 atom, or a concrete VRAM budget spreadsheet for Qwen3-8B at various context lengths and precisions.
