# Qwen3 SM89 CuTeDSL Strategy, Iteration Log, Speedups, and Research Reference

This document consolidates the root-level project notes for the Qwen3-8B on RTX 4080 Laptop work into a single, indexable reference.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Hardware Target](#hardware-target)
3. [Repository Workflow](#repository-workflow)
4. [Kernel Build Order](#kernel-build-order)
5. [Iteration Log Template](#iteration-log-template)
6. [Speedup Reference](#speedup-reference)
7. [Research Notes and Key Findings](#research-notes-and-key-findings)
8. [Profiling and Benchmarking Checklist](#profiling-and-benchmarking-checklist)
9. [Suggested Next Steps](#suggested-next-steps)

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

The speedup reference is meant to keep the project grounded in the broader ecosystem.

### RMSNorm
- Fused RMSNorm and RoPE patterns can be much faster than naive split kernels.
- In practice, the win mostly comes from reducing HBM traffic and launches.

### RoPE
- A fused RoPE path can outperform a naive PyTorch baseline by a large margin.
- The main issue is reducing intermediate allocations and kernel launch overhead.

### GEMM
- Shape-specialized kernels can outperform generic kernels.
- On SM89, careful tile size selection and swizzles matter a lot.

### Attention
- Attention at decode is often constrained by KV-cache traffic rather than arithmetic alone.
- Flash-style layouts and split-KV designs are particularly relevant.

### System-level wins
- CUDA graphs can reduce launch overhead for repeated decode steps.
- Speculative decoding is a strong end-to-end optimization once kernel-level bottlenecks are reduced.

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
