# SM89 Ada MMA Reading Guide

You validated that your GPU is `sm_89`, which means the relevant Tensor Core
programming model is Ada/Ampere-style warp-level `mma.sync`.

The practical path for this GPU is:

```text
global memory
-> coalesced/vectorized loads or cp.async staging
-> shared memory layout chosen by CuTe/CUTLASS
-> ldmatrix from shared memory into registers
-> mma.sync on Tensor Cores
-> accumulator registers
-> global memory store
```

The non-relevant path for this GPU is:

```text
TMA / CUtensorMap / wgmma.mma_async / tcgen05 / TMEM
```

Those are Hopper or Blackwell-family concepts, not Ada `sm_89`.

## Best Reading Order

| Order | Resource | What To Read For | Why It Matters For SM89 | What You Should Do With It |
|---:|---|---|---|---|
| 1 | [Spatters: Implementing a fast Tensor Core matmul on Ada](https://www.spatters.ca/mma-matmul) | Ada-specific `mma`, `ldmatrix`, `cp.async`, permuted shared memory, and the progression from naive MMA to near-cuBLAS performance. | This is the most directly relevant public walkthrough for your `sm_89` GPU. It uses an Ada GPU, chooses the PTX MMA path, and explicitly rejects WGMMA for Ada. The key quote is: "`wgmma is not an option`" for the Ada GPU used in the post. | Treat this as the practical performance ladder: first make `mma.sync` correct, then fix global loads, then fix shared-memory layout, then pipeline global-to-shared copies. |
| 2 | [Lei Mao: NVIDIA Tensor Core MMA Instruction TN Layout](https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/) | Why Tensor Core operand layout is not cosmetic. | The post says newer architectures are "specific to TN layout" for most MMA instructions and reports TN far ahead in a benchmark of layout families. That tells you why layout choice is part of the instruction contract, not just a Python-level tensor convention. | When you see CuTe layouts for A/B, ask: "does this eventually feed the row/col contract of the selected MMA atom without extra shuffling or bank conflicts?" |
| 3 | [Lei Mao: CuTe ldmatrix](https://leimao.github.io/blog/CuTe-ldmatrix/) | The shared-memory-to-register path that feeds MMA fragments. | The key quote is that `ldmatrix` loads matrices "from shared memory to registers for `mma`". On SM89, that is exactly the missing link between a CuTe shared-memory `Layout` and the per-lane register fragments consumed by `mma.sync`. | Drill into `make_tiled_copy_A/B`, `LdMatrix8x8x16bOp`, and `partition_S/D`. Those tell you how the CuTe layout becomes the register fragment layout. |
| 4 | [A gentle introduction to GEMM using MMA tensor cores](https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/) | A slower conceptual bridge from scalar GEMM to warp-level MMA. | It explains why MMA is warp-level and why WMMA is easier but hides important layout details. That is useful before reading CuTe code where the same concepts become `TiledMma`, fragments, and per-thread partitions. | Use this before trying to understand every CuTe type. First understand one warp computing one `m16n8k16` tile. |
| 5 | [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) | Official occupancy and memory limits for Ada. | It gives the hardware limits that bound your tile search: 48 resident warps/SM, 24 resident blocks/SM, 64K 32-bit registers/SM, 100 KB shared memory/SM, and 99 KB shared memory/block. | Use these numbers to decide whether a tile shape can have enough resident CTAs/warps after registers and shared memory are counted. |
| 6 | [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) | The official instruction contracts and target architecture requirements. | This is where you verify whether an instruction family is legal on SM89. For example, PTX lists `wgmma.mma_async` as requiring `sm_90a`, and bulk tensor async copy as requiring `sm_90` or higher. | Use PTX as the final arbiter when a blog mentions Hopper, Blackwell, tensor maps, or descriptor swizzles. If the target says `sm_90+`, it is not your Ada path. |
| 7 | [Colfax: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) | What WGMMA descriptors and warpgroup layouts are for. | This is valuable as contrast. It explains Hopper WGMMA as a 128-thread warpgroup operation, which is a different programming model from Ada warp-level `mma.sync`. | Read it to avoid mixing generations: descriptors and WGMMA layouts are worth learning later, but they should not drive your SM89 implementation. |
| 8 | [Lei Mao: Benchmarking Tensor Core MMA Peak Performance](https://leimao.github.io/blog/Benchmarking-NVIDIA-Tensor-Core-MMA-Peak-Performances/) | How to benchmark MMA directly instead of relying on theoretical peak. | It reinforces the habit of measuring the instruction family and shape you actually use. For SM89, that means profiling `mma.sync` kernels, not extrapolating from WGMMA/TMA examples. | After correctness, run Nsight Compute and compare tile sweeps by tensor pipe utilization, memory stalls, and bank conflicts. |
| 9 | [NVIDIA CUTLASS Documentation](https://docs.nvidia.com/cutlass/) and this repo's [tensorop_gemm.py](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py) | Production-style CuTe structure: `TiledMma`, layouts, copy atoms, fragments, and pipelined mainloop. | The local example is close to your GPU generation: it uses `MmaF16BF16Op`, `LdMatrix8x8x16bOp`, `cp.async`, a `128x128x32` CTA tile, and 128 threads by default. | Use it as the concrete SM89 template after the conceptual reading. |

## Source Evidence Snapshot

| Source | Short Cited Quote | What The Quote Supports |
|---|---|---|
| [Spatters Ada MMA](https://www.spatters.ca/mma-matmul) | "`wgmma is not an option`" | Ada should be learned through `mma.sync`, not Hopper WGMMA. |
| [Spatters Ada MMA](https://www.spatters.ca/mma-matmul) | "`m16n8k16`" | The practical FP16/BF16 Ada MMA atom to study first. |
| [Lei Mao TN Layout](https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/) | "`specific to TN layout`" | Operand layout is part of Tensor Core performance, not just naming. |
| [Lei Mao CuTe ldmatrix](https://leimao.github.io/blog/CuTe-ldmatrix/) | "from shared memory to registers for `mma`" | `ldmatrix` is the bridge between CuTe shared-memory layouts and MMA fragments. |
| [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) | "`48`" resident warps/SM | Occupancy and CTA-size decisions must fit Ada's official SM limits. |
| [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) | "`sm_90a`" | WGMMA is outside the `sm_89` execution path. |
| [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html) | "`sm_90` or higher" | TMA/tensor async copy is outside the `sm_89` execution path. |
| [Colfax WGMMA Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/) | "`four contiguous warps`" | WGMMA is a warpgroup model, while SM89 `mma.sync` is a warp-level model. |

## SM89 Starting Choices With Validation

| Aspect | First Choice | Validation | Source |
|---|---|---|---|
| Instruction family | `mma.sync`, not WGMMA | Your GPU is `sm_89`. PTX marks WGMMA as Hopper-class, requiring `sm_90a`, so it is not the right execution path for Ada. | [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [Spatters Ada MMA](https://www.spatters.ca/mma-matmul) |
| FP16/BF16 atom | `m16n8k16` | This is the common Ada/Ampere FP16/BF16 warp-level Tensor Core MMA shape. The local CuTe Tensor Core GEMM example also sets `mma_inst_shape = (16, 8, 16)`. | [tensorop_gemm.py:116](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:116), [Spatters Ada MMA](https://www.spatters.ca/mma-matmul), [A gentle intro to MMA](https://am17an.bearblog.dev/a-gentle-introduction-to-gemm-using-mma-tensor-cores/) |
| Operand layout family | TN-style first: A M-major/row-style, B N-major in the repo's `(N,K)` representation | Lei Mao's TN-layout article argues that newer Tensor Core MMA instruction families are primarily optimized around TN. The local Tensor Core example defaults to `--a_major m --b_major n`, matching this starting point after accounting for its B-as-`N x K` convention. | [Lei Mao TN Layout](https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/), [tensorop_gemm.py:902](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:902), [tensorop_gemm.py:995](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:995) |
| CTA tile | `128x128x32` | This is the default CTA tile in the local CuTe Tensor Core GEMM example. It gives a serious baseline tile with reuse across both M and N before sweeping smaller/asymmetric tiles. | [tensorop_gemm.py:109](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:109) |
| Atom layout | `2,2,1` | This gives 4 warp-level MMA atoms per CTA: 2 across M, 2 across N, 1 across K. It matches the local example's default logic and creates a 4-warp CTA baseline. | [tensorop_gemm.py:104](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:104), [tensorop_gemm.py:231](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:231) |
| Threads/CTA | `128` | The local Tensor Core example computes `num_threads = atom_layout_M * atom_layout_N * atom_layout_K * 32`. With `2,2,1`, that is `2 * 2 * 1 * 32 = 128`. | [tensorop_gemm.py:113](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:113) |
| Pipeline stages | `3` | The local Tensor Core example defaults to `num_stages = 3` and asserts at least 3 stages. This gives a first triple-buffered global-to-shared pipeline. | [tensorop_gemm.py:110](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:110), [tensorop_gemm.py:127](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:127) |
| SMEM to registers | `LdMatrix8x8x16bOp` | `ldmatrix` is the Ada/Ampere path for loading 16-bit matrix fragments from shared memory into registers for `mma.sync`. The local Tensor Core example creates `LdMatrix8x8x16bOp` copy atoms for A and B. | [Lei Mao CuTe ldmatrix](https://leimao.github.io/blog/CuTe-ldmatrix/), [tensorop_gemm.py:544](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:544) |
| Data movement | `cp.async` / `CopyG2SOp`, not TMA | Ada uses `cp.async`-style global-to-shared staging. PTX marks bulk tensor async copy/TMA-style mechanisms as `sm_90+`, so TMA is not the SM89 path. | [NVIDIA PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html), [CUDA async copy docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), [tensorop_gemm.py:544](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:544) |

## Search Space After The First Baseline

After the first working SM89 Tensor Core baseline, do not randomly change all
parameters. Change one CuTe-level decision at a time, then measure. The point
of using CuTe DSL is that the search space is expressed as layouts, tilers,
copy atoms, and tiled MMA objects rather than raw pointer arithmetic.

Baseline to hold fixed for iteration 0:

```text
Instruction family: mma.sync
MMA atom:           m16n8k16
Operand layout:     TN-style: a_major=m, b_major=n
CTA tile:           128x128x32
atom_layout_mnk:    2,2,1
threads/CTA:        128
stages:             3
SMEM->register:     LdMatrix8x8x16bOp
GMEM->SMEM:         cp.async / CopyG2SOp
```

| Iteration | Search Space | CuTe DSL Lever | Why Try It | Validation Signal | Resource |
|---:|---|---|---|---|---|
| 1 | CTA tile aspect ratio: `128x128x32`, `128x64x32`, `64x128x32`, `64x64x32` | `self.cta_tiler`, `cute.local_tile`, CTA grid shape | Transformer GEMMs are often rectangular. Different M/N shapes change data reuse, register pressure, and occupancy. | Runtime, TFLOP/s, achieved occupancy, DRAM throughput, tensor pipe utilization | [tensorop_gemm.py:109](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:109), [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) |
| 2 | Warp/MMA atom placement: `2,2,1`, then M-heavy or N-heavy variants such as `4,1,1` and `1,4,1` if the code supports them | `atom_layout_mnk`, `cute.make_layout`, `cute.make_tiled_mma` | This changes how warps are distributed across the CTA output tile. It tests whether the problem benefits from more parallelism along M or N. | Eligible warps per scheduler, tensor pipe issue rate, register use, correctness | [tensorop_gemm.py:104](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:104), [CUTLASS CuTe MMA Atom docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html), [Lei Mao CuTe Tiled MMA](https://leimao.github.io/blog/CuTe-Tiled-MMA/) |
| 3 | Operand layout family: keep TN-style first, then test non-TN input conventions only through CuTe layout/copy transformations | `a_major`, `b_major`, `LayoutEnum.from_tensor`, `LdMatrix8x8x16bOp(trans=...)`, `make_tiled_copy_A/B` | Lei Mao's TN article says newer Tensor Core MMA families are built around TN. On SM89, the search should optimize TN first, then express other user-facing layouts by changing CuTe layouts/copies rather than expecting separate non-TN MMA atoms. | Correctness, tensor pipe utilization, extra copy/transposition cost, shared-memory bank conflicts | [Lei Mao TN Layout](https://leimao.github.io/blog/NVIDIA-Tensor-Core-MMA-Instruction-TN-Layout/), [tensorop_gemm.py:141](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:141), [tensorop_gemm.py:546](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:546) |
| 4 | Shared-memory A/B layout and swizzle choices | `cute.make_layout`, `cute.make_composed_layout`, swizzle/layout modes used for `sA` and `sB` | On Ada, the WGMMA descriptor/tensor-map swizzle path is unavailable. The equivalent search is software-controlled CuTe shared-memory layout that feeds `ldmatrix` with fewer bank conflicts. | Shared-memory bank conflicts, SMEM throughput, tensor pipe stalls waiting on data | [Lei Mao CuTe ldmatrix](https://leimao.github.io/blog/CuTe-ldmatrix/), [CuTe Layout Representation and Algebra](https://arxiv.org/abs/2603.02298), [tensorop_gemm.py:544](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:544) |
| 5 | `ldmatrix` orientation and grouping: normal vs transposed, `.x1/.x2/.x4`-equivalent choices exposed by CuTe copy atoms | `cute.nvgpu.warp.LdMatrix8x8x16bOp`, `make_tiled_copy_A`, `make_tiled_copy_B` | The copy atom must match the MMA fragment layout and the SMEM layout. Wrong orientation can produce wrong fragments; smaller groupings can cost extra instructions. | Correctness first, then SMEM-to-register instruction count, bank conflicts, tensor utilization | [Lei Mao CuTe ldmatrix](https://leimao.github.io/blog/CuTe-ldmatrix/), [CuTe DSL API: `make_tiled_copy_A/B`](https://docs.nvidia.com/cutlass/latest/media/docs/pythonDSL/cute_dsl_api/cute.html), [tensorop_gemm.py:560](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:560) |
| 6 | Global-to-shared copy vectorization and alignment | `cute.nvgpu.cpasync.CopyG2SOp`, `cute.make_tiled_copy_tv`, vector width / `num_bits_per_copy`, tensor alignment | A good Tensor Core mainloop can still starve if GMEM copies are uncoalesced or too narrow. This iteration tunes how the CTA fills SMEM. | DRAM throughput, L2 hit rate, copy stalls, tensor pipe starvation | [CUDA async copy docs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html), [sgemm.py:128](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/sgemm.py:128), [Spatters Ada MMA](https://www.spatters.ca/mma-matmul) |
| 7 | Pipeline depth: `3` vs `4` stages | `self.num_stages`, SMEM allocation shape, `cp_async_commit_group`, `cp_async_wait_group` | More stages can hide more GMEM latency, but consume more shared memory and can reduce resident CTAs. | Long scoreboard stalls, achieved occupancy, shared memory per block, runtime | [tensorop_gemm.py:110](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:110), [tensorop_gemm.py:524](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:524), [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) |
| 8 | K tile depth: keep `32`, then consider `64` only if layout, SMEM, and pipeline constraints remain clean | `cta_tiler` K dimension, `local_tile`, mainloop K blocking | Larger K blocks increase reuse per CTA but increase SMEM footprint and can reduce occupancy. | Occupancy, SMEM per block, tensor pipe utilization, runtime across large K | [tensorop_gemm.py:109](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:109), [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html) |
| 9 | Output/epilogue store layout | `partition_C`, accumulator fragment layout, shared-memory epilogue staging, global store layout | Even if MMA is fast, bad stores can leave performance on the table. Epilogue layout decides whether writes to C are coalesced and whether extra staging is needed. | Store efficiency, DRAM write throughput, runtime, correctness | [tensorop_gemm.py:529](/home/lily/wsl_git/cutelearning/cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py:529), [CUTLASS CuTe MMA Atom docs](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0t_mma_atom.html) |
| 10 | Problem-shape sweep: square and transformer-shaped GEMMs | Host launcher parameters, grid shape, same kernel variants | A tile that wins on `4096x4096x4096` may not win for QKV, MLP up/down, or attention score shapes. | Per-shape TFLOP/s, kernel time, occupancy, tensor utilization | PyTorch/cuBLAS baseline, Nsight Compute, [Spatters Ada MMA](https://www.spatters.ca/mma-matmul) |

The most important CuTe-specific loop is iterations 2 through 5:

```text
TiledMma decides the per-thread fragment ownership.
Operand layout decides which A/B major modes the MMA path is optimized around.
Shared-memory Layout decides where the tile physically lives.
LdMatrix copy atom decides how a warp moves that layout into registers.
make_tiled_copy_A/B(..., tiled_mma) checks that copy and MMA agree.
```

If one of those four pieces changes, re-check the other three. That is the
core CuTe DSL advantage: the layout search is explicit and inspectable rather
than hidden in hand-written pointer arithmetic.

## Roofline And Bottleneck Analysis For The Search Space

Roofline analysis gives the search space a concrete purpose. Each parameter
change should answer one of these questions:

```text
Is this kernel limited by Tensor Core math?
Is it limited by global memory bandwidth?
Is it limited by shared memory / ldmatrix?
Is it limited by scheduling/occupancy?
Is it limited by instruction overhead?
Is it limited by stores in the epilogue?
```

The high-level roofline idea is:

```text
achievable FLOP/s = min(peak compute FLOP/s, arithmetic intensity * memory bandwidth)
```

where:

```text
arithmetic intensity = FLOPs / bytes moved
```

For GEMM:

```text
FLOPs = 2 * M * N * K
```

For FP16/BF16 inputs and FP32 output, a rough full-GEMM lower-bound byte model is:

```text
bytes = 2*M*K + 2*K*N + 4*M*N
```

So:

```text
AI = 2*M*N*K / (2*M*K + 2*K*N + 4*M*N)
```

For `4096,4096,4096`:

```text
FLOPs = 2 * 4096^3
      = 137,438,953,472

Bytes = 2*4096*4096 + 2*4096*4096 + 4*4096*4096
      = 134,217,728

AI = 1024 FLOP/byte
```

That arithmetic intensity is high. A good large square Tensor Core GEMM should
usually be compute/Tensor-Core limited, not DRAM-bandwidth limited. If a
`4096^3` custom GEMM is DRAM-limited, the kernel is probably failing to reuse
A/B tiles correctly or failing to keep the Tensor Core pipe fed.

You can also estimate the local CTA mainloop intensity. For a `128x128x32`
CTA tile:

```text
FLOPs per K tile = 2 * 128 * 128 * 32
                 = 1,048,576

A bytes = 128 * 32 * 2
        = 8,192

B bytes = 32 * 128 * 2
        = 8,192

A+B bytes = 16,384

Mainloop AI = 1,048,576 / 16,384
            = 64 FLOP/byte
```

Do not count the C write once per K tile. C is written after the full reduction
over K, so include C in full-GEMM analysis or epilogue analysis, not in every
mainloop K stage.

### Useful Roofline Workflow For SM89 GEMM

| Search Parameter | Roofline Question | What To Compute / Measure | Method To Extract Profiling | Interpretation |
|---|---|---|---|---|
| CTA tile shape | Does this tile have enough data reuse? | Arithmetic intensity, runtime, TFLOP/s, DRAM throughput | Python/PyTorch timing for runtime; formula for AI; `ncu --set full` for DRAM and tensor metrics | Larger tiles usually raise reuse. If performance rises with tile size, the previous shape may have been memory-limited or overhead-heavy. |
| K tile depth | Does deeper K blocking improve reuse or just reduce occupancy? | FLOPs per CTA, SMEM bytes per CTA, resident CTAs/SM | Python formula for FLOPs/bytes; CuTe compile-time tile parameters; Nsight Compute Occupancy section | Larger K can improve pipeline efficiency but may consume more shared memory and lower resident CTAs. |
| Atom layout | Are warps issuing enough MMA work? | Tensor pipe utilization, eligible warps per scheduler, register use | Nsight Compute `smsp__pipe_tensor*`, scheduler stats, launch/resource report | Low tensor utilization means the MMA pipe is not fed, even if the kernel has high occupancy. |
| Threads/CTA | Are there enough active warps to hide latency? | Achieved occupancy, active warps/SM, eligible warps/scheduler | Nsight Compute Occupancy and Scheduler Statistics sections | Higher occupancy helps only if the stalls are latency-related. Maximum occupancy is not automatically maximum performance. |
| Pipeline stages | Is GMEM latency hidden? | Long scoreboard stalls, async copy wait behavior, tensor pipe idle time | Nsight Compute stall metrics; inspect `cp_async_wait_group` placement in CuTe code | If more stages reduce long scoreboard stalls and improve tensor utilization, the previous pipeline was too shallow. |
| Shared-memory layout | Is SMEM becoming the bottleneck? | Shared-memory bank conflicts, SMEM throughput, ldmatrix-related stalls | Nsight Compute shared-memory tables; compare CuTe layout/swizzle variants | Bank conflicts mean the CuTe SMEM layout and warp access pattern disagree. |
| `ldmatrix` copy atom | Is SMEM-to-register movement efficient and correct? | Shared-load instruction count, bank conflicts, correctness, tensor utilization | Nsight Compute SASS/instruction stats; correctness check against PyTorch; CuTe `make_tiled_copy_A/B` usage | Wrong orientation or grouping can produce wrong fragments, extra instructions, or low Tensor Core utilization. |
| GMEM copy vectorization | Are global loads coalesced and wide enough? | DRAM throughput, L2 throughput, global load efficiency, memory stalls | Nsight Compute Memory Workload Analysis; inspect `CopyG2SOp` and `num_bits_per_copy` | Low bandwidth with memory stalls often means bad copy layout, poor alignment, or non-coalesced loads. |
| Epilogue/store layout | Are writes to C wasting bandwidth? | Global store efficiency, DRAM write throughput, store instruction count | Nsight Compute Memory Workload Analysis; inspect `partition_C` and epilogue store path | Bad stores can dominate smaller or skinny GEMMs even when the mainloop is good. |
| Problem shape | Does the same kernel work for square and transformer-shaped GEMMs? | TFLOP/s per shape, Tensor Core utilization, DRAM throughput, occupancy | Python benchmark loop over shapes; PyTorch/cuBLAS baseline; Nsight Compute on representative shapes | A tile that wins on square GEMM may lose on QKV, MLP up/down, or attention-score shapes. |

### Practical Profiling Commands

Use Python or PyTorch timing first because it is fast:

```bash
python learning/leetgpu/naive_gemm_pytorch.py --mnk 1024,1024,1024
python learning/leetgpu/naive_gemm_pytorch.py --mnk 2048,2048,2048
python learning/leetgpu/naive_gemm_pytorch.py --mnk 4096,4096,4096
```

Then use Nsight Compute when a kernel is correct and worth inspecting:

```bash
ncu --set full python learning/leetgpu/naive_gemm_pytorch.py --mnk 4096,4096,4096
```

For a lighter first pass:

```bash
ncu --section SpeedOfLight \
    --section MemoryWorkloadAnalysis \
    --section SchedulerStats \
    python learning/leetgpu/naive_gemm_pytorch.py --mnk 4096,4096,4096
```

Use a PyTorch/cuBLAS-style baseline for comparison:

```python
import torch

M, K, N = 4096, 4096, 4096
A = torch.randn((M, K), device="cuda", dtype=torch.float16)
B = torch.randn((K, N), device="cuda", dtype=torch.float16)

torch.cuda.synchronize()
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
C = A @ B
end.record()
torch.cuda.synchronize()

ms = start.elapsed_time(end)
tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12
print("ms:", ms)
print("TFLOP/s:", tflops)
```

### Nsight Compute Signals To Watch

The exact metric names can vary by Nsight Compute version, but these categories
are the ones to inspect:

```text
SpeedOfLight / Roofline
SM throughput
Tensor pipe utilization
DRAM throughput
L2 throughput
Shared-memory bank conflicts
Shared-memory throughput
Active warps per SM
Eligible warps per scheduler
Long scoreboard stalls
Short scoreboard stalls
MIO throttle stalls
HMMA / MMA instruction count
Global load/store efficiency
```

Useful metric-name families to search for in Nsight Compute:

```text
sm__throughput
smsp__pipe_tensor
smsp__warps_active
smsp__warps_eligible
smsp__stall_long_scoreboard
smsp__stall_short_scoreboard
smsp__stall_mio_throttle
dram__throughput
lts__throughput
l1tex__data_bank_conflicts
sass_thread_inst_executed_op_hmma
```

### Bottleneck Classification

| Observation | Likely Bottleneck | Next Search Direction |
|---|---|---|
| High DRAM throughput, low tensor utilization | Not enough reuse or GMEM bottleneck | Try larger CTA tiles, better GMEM copy vectorization, or more pipeline stages. |
| Low DRAM throughput, low tensor utilization, high long scoreboard stalls | Latency is not hidden | Try more stages, more resident warps, or smaller shared-memory footprint. |
| High shared-memory bank conflicts | Bad SMEM layout for `ldmatrix` | Change CuTe shared-memory layout/swizzle and re-check `make_tiled_copy_A/B`. |
| Low eligible warps per scheduler | Not enough independent ready work | Try more warps/CTA, more resident CTAs, or reduce dependency chains. |
| High occupancy but low tensor utilization | Occupancy is not the real problem | Fix data movement, layout, copy atom, or MMA issue pattern. |
| Good tensor utilization but slow stores | Epilogue/store bottleneck | Improve C layout/store coalescing or epilogue staging. |
| Fast square GEMM but slow skinny GEMM | Tile shape mismatch | Sweep asymmetric CTA tiles and transformer-shaped problem sizes. |

### Experiments That Match The Search Space

| Experiment | What It Tests |
|---|---|
| Sweep `64x64x32`, `128x64x32`, `64x128x32`, `128x128x32` | Whether larger CTA tiles improve reuse enough to offset occupancy/register pressure. |
| Sweep K tile `32` vs `64` | Whether deeper K staging improves math-pipe feeding or hurts occupancy. |
| Sweep `3` vs `4` stages | Whether global-memory latency is visible in the timeline. |
| Sweep `atom_layout_mnk` | Whether warp distribution across M/N changes tensor pipe utilization. |
| Compare scalar loads vs `ldmatrix` path | Isolate the value of `ldmatrix` and layout-matched shared-memory loads. |
| Compare naive SMEM layout vs swizzled CuTe layout | Directly expose shared-memory bank conflict cost. |
| Compare custom kernel vs PyTorch `A @ B` | Establish the practical cuBLAS/cuBLASLt baseline. |

The key interpretation rule is:

```text
Each parameter changes either:
1. arithmetic intensity,
2. memory bandwidth efficiency,
3. latency hiding,
4. tensor issue efficiency,
5. shared-memory conflict behavior,
6. or store efficiency.
```

That framing keeps the CuTe DSL search from becoming arbitrary. You are not
just trying tile shapes. You are testing which roofline ceiling or stall source
currently controls the kernel.

### Blog-Ready Profiling And Visualization Plan

If the profiling work will become a Substack or technical blog post, change the
workflow slightly. Engineering profiling can be messy and exploratory. A public
technical writeup needs each experiment to be reproducible, visually clear, and
attached to one claim.

The reader-facing story should be:

```text
1. Establish the hardware and baseline.
2. Show the first correct CuTe/CUTLASS Tensor Core baseline.
3. Change one search-space parameter at a time.
4. Show which roofline ceiling or stall reason moved.
5. Explain the next decision using the measurement.
```

Do not publish only final TFLOP/s numbers. A learning-focused post should show
why the search moved from one design to the next.

| Blog Figure | What It Shows | Data To Collect | How To Collect | Why Readers Care |
|---|---|---|---|---|
| Hardware summary table | SM89 limits and test environment | GPU name, driver, CUDA version, compute capability, memory size, max warps/SM, shared memory/SM | `nvidia-smi`, `torch.cuda.get_device_name`, CUDA device properties, Ada tuning guide | Makes results reproducible and prevents Blackwell/Hopper advice from being confused with Ada advice. |
| Baseline comparison bar chart | PyTorch/cuBLAS vs first custom CuTe kernel | Runtime, TFLOP/s, max error | Python timing with CUDA events; correctness check against PyTorch | Establishes the performance gap and proves the kernel is numerically valid. |
| Roofline scatter plot | Whether variants are memory-bound or compute-bound | Arithmetic intensity, achieved TFLOP/s, theoretical memory roof, practical baseline roof | Python formula for AI; Python timing; hardware bandwidth estimate; optional Nsight Compute roofline | Gives a visual answer to "is this tile shape limited by memory or math?" |
| CTA tile sweep line/bar chart | Effect of `64x64`, `128x64`, `64x128`, `128x128` | TFLOP/s, occupancy, DRAM throughput, tensor utilization | Python benchmark loop; `ncu` on representative winners/losers | Shows why the tile search is not arbitrary. |
| Layout/bank-conflict chart | Effect of SMEM layout/swizzle choices | Shared-memory bank conflicts, SMEM throughput, tensor utilization | Nsight Compute shared-memory metrics | Makes the `ldmatrix`/CuTe-layout discussion concrete. |
| Pipeline-stage chart | Whether 3 or 4 stages hides latency better | Runtime, long scoreboard stalls, occupancy, shared memory/block | `ncu` stall metrics and occupancy section | Shows the latency-hiding tradeoff against shared-memory footprint. |
| Tensor utilization chart | Whether MMA issue rate improved | Tensor pipe utilization, eligible warps/scheduler, HMMA/MMA instruction count | Nsight Compute SpeedOfLight and scheduler metrics | Separates "more occupancy" from "more useful Tensor Core work." |
| Shape generalization table | Whether the result works for LLM-like shapes | Runtime and TFLOP/s for square, MLP up/down, QK, PV shapes | Python benchmark loop over shapes; PyTorch baseline | Prevents overfitting the blog to only `4096^3`. |
| Final decision table | Which parameter choices survived | Best tile, atom layout, stages, layout family, copy path | Summarize from benchmark CSV | Gives readers a compact recipe they can try. |

For the blog, record every run into a CSV, even if the first version is manual:

```text
timestamp
gpu
driver
cuda_version
problem_m
problem_n
problem_k
dtype_a
dtype_b
dtype_c
kernel_name
cta_m
cta_n
cta_k
atom_layout_m
atom_layout_n
atom_layout_k
threads_per_cta
stages
operand_layout
smem_layout_label
ldmatrix_variant
runtime_ms
tflops
max_error
achieved_occupancy
tensor_pipe_utilization
dram_throughput
l2_throughput
shared_bank_conflicts
eligible_warps_per_scheduler
long_scoreboard_stall_pct
notes
```

This CSV becomes the source for every chart. Do not hand-copy chart values from
terminal output into a blog draft.

Recommended visual order:

```text
Figure 1: Hardware and kernel data path diagram
Figure 2: PyTorch/cuBLAS baseline vs first CuTe baseline
Figure 3: Roofline scatter for the first tile sweep
Figure 4: CTA tile sweep with tensor utilization overlay
Figure 5: Shared-memory bank conflicts before/after layout change
Figure 6: Pipeline stages vs stalls and occupancy
Figure 7: Transformer-shaped GEMM table
Figure 8: Final search-space decision table
```

For blog credibility, include these rules in the methodology:

| Rule | Reason |
|---|---|
| Warm up before timing | Avoid one-time compilation, cache, and clock ramp effects. |
| Use CUDA events for kernel timing | Python wall-clock timing includes host overhead. |
| Synchronize before reading timings | CUDA launches are asynchronous. |
| Report median and spread | A single best run hides variance. |
| Compare against PyTorch `A @ B` | Gives readers a familiar cuBLAS/cuBLASLt baseline. |
| Keep problem shapes fixed across variants | Avoid comparing different amounts of work. |
| Change one parameter per chart | Makes causal interpretation possible. |
| Keep correctness checks in the loop | Fast wrong kernels are common when changing layouts. |
| Archive exact command lines | Makes the blog reproducible. |

For a learning blog, the most valuable charts are not necessarily the prettiest.
The strongest visuals are the ones that make a hidden hardware constraint
visible:

```text
TN / row.col layout       -> fewer layout conversions and a cleaner MMA path
CuTe SMEM swizzle         -> fewer bank conflicts
LdMatrix copy atom        -> correct fragment loading into registers
More pipeline stages      -> fewer long scoreboard stalls, until occupancy drops
Larger CTA tile           -> more reuse, until registers/SMEM reduce occupancy
```

Avoid presenting `4096,4096,4096` as the only truth. Use it as the clean
roofline/profiling size, then add transformer-shaped cases:

```text
Square GEMM:       4096 x 4096 x 4096
MLP up projection: 1024 x 1024 x 4096
MLP down project:  1024 x 4096 x 1024
QK^T score:        1024 x 128 x 1024
PV output:         1024 x 1024 x 128
```

The final blog claim should be modest and specific:

```text
On SM89, this search space shows how CuTe layouts, ldmatrix-compatible shared
memory, cp.async staging, and mma.sync atom choices affect a learning GEMM.
It is not a claim that one hand-written kernel universally beats cuBLAS.
```

## Evidence-Based Decisions

### Why `mma.sync` Is The First-Class Path On SM89

Your GPU is `sm_89`, so the Tensor Core instruction family to study first is
warp-level `mma.sync`. The Ada-specific blog is explicit that its kernels use
the PTX MMA API and not WGMMA because the target is Ada. NVIDIA's PTX ISA gives
the formal reason: `wgmma.mma_async` requires `sm_90a`, while your device is
`sm_89`.

That means the design center is:

```text
32-thread warp
-> per-lane A/B/C fragments
-> mma.sync.aligned.m16n8k16.row.col...
-> per-lane accumulator fragments
```

not:

```text
128-thread warpgroup
-> WGMMA matrix descriptors
-> wgmma.mma_async
```

The Colfax Hopper guide is still worth reading, but mainly to understand what
you are not doing on Ada. Hopper WGMMA is a warpgroup operation: 4 contiguous
warps, or 128 threads, execute one async matrix operation together. Ada
`mma.sync` is one warp at a time.

### Why `ldmatrix`-Compatible Layouts Matter

#### What `ldmatrix` Actually Does

The PTX ISA description is the authoritative starting point:

> *"The `ldmatrix` instruction performs a matrix load operation from shared memory,
> loading matrices of 8×8 bytes from shared memory into registers.
> This instruction is designed to be used in combination with tensor core
> matrix multiply-accumulate operations."*
> — [NVIDIA PTX ISA: ldmatrix](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-load-instruction-ldmatrix)

The full instruction name is `ldmatrix.sync.aligned.x4.m8n8.shared.b16`.
- `x4`: load 4 sub-matrices per warp (one 16×16 chunk of 16-bit elements)
- `m8n8`: each sub-matrix is 8×8 half-precision values
- `shared`: source is shared memory
- Each thread in the warp contributes one source address and receives data
  into two 32-bit registers

Lei Mao's CuTe ldmatrix article is the clearest explanation of why this matters:

> *"The `ldmatrix` instruction loads matrices from shared memory to registers
> for `mma`. Without `ldmatrix`, the programmer must manually load the correct
> elements into each thread's registers to match the fragment layout required
> by `mma`."*
> — [Lei Mao: CuTe ldmatrix](https://leimao.github.io/blog/CuTe-ldmatrix/)

#### Why Scalar Loads Are Worse Than `ldmatrix`

The alternative — each thread issuing scalar `ld.shared` instructions — has
three specific failures:

**1. Bank conflicts without swizzle.**
Shared memory has 32 banks, each 4 bytes wide. For a 128-wide row of FP16
values, consecutive threads reading consecutive 2-byte elements hit banks
0, 0, 1, 1, 2, 2, ... — two threads per bank, 16-way conflict across the warp.
The Spatters Ada MMA blog quantifies this:

> *"The permuted shared memory layout is essential for avoiding bank conflicts
> when loading to registers. Without it, each load has 16-way bank conflicts,
> reducing effective bandwidth by 16x."*
> — [Spatters: Ada MMA matmul](https://www.spatters.ca/mma-matmul)

**2. Wrong element-to-thread mapping.**
`mma.sync` expects specific elements in specific threads' registers. The PTX ISA
specifies exactly which thread holds which row/column for the m16n8k16 atom.
Scalar loads give each thread a contiguous chunk of the matrix; that is the
wrong shape. You would need extra shuffles (`shfl.sync`) to rearrange, which
costs cycles and extra instructions.

**3. No warp-level hardware gather.**
`ldmatrix` uses a hardware scatter-gather that reads 8 non-contiguous addresses
(one per row of the 8×8 sub-tile) in one instruction, using each thread's
contributed address. Scalar loads can only read one address per instruction.
Loading an 8×8 FP16 sub-tile with scalar loads takes 8 instructions per thread;
`ldmatrix.x4` does it in 1 instruction for the full warp.

#### The Three Contracts That Must Agree

That is why "ldmatrix-compatible layout" means more than "the dimensions line
up." It means these three contracts agree:

| Contract | CuTe Object | What Would Break If Wrong |
|---|---|---|
| Shared-memory address mapping | `Layout`, `ComposedLayout`, swizzle | Without swizzle, neighboring lanes hit the same bank: 16-way conflict on every load. |
| Shared-to-register movement | `LdMatrix8x8x16bOp`, `make_tiled_copy_A/B` | If the copy atom does not match MMA's fragment shape, elements land in wrong registers. |
| Register fragment use | `TiledMma`, `partition_A/B/C`, `make_fragment_A/B/C` | Each thread must own the exact rows/columns the `mma.sync` atom expects; wrong ownership = silent wrong results. |

#### How CuTeDSL Connects `ldmatrix` to Its Layouts

In CuTeDSL, `ldmatrix` is wrapped as a copy atom. Lei Mao's article describes
the connection:

> *"CuTe provides `SM75_U32x4_LDSM_N` (and related aliases) as copy atoms that
> emit `ldmatrix` under the hood. The key is that `make_tiled_copy_A(atom, tiled_mma)`
> derives a tiled copy whose layout is consistent with the tiled MMA's A-fragment
> layout."*
> — [Lei Mao: CuTe ldmatrix](https://leimao.github.io/blog/CuTe-ldmatrix/)

In the local `tensorop_gemm.py`, this connection is explicit:

| Code Location | What It Shows |
|---|---|
| [tensorop_gemm.py:116](../../../cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py#L116) | `MmaAtomShape = (16, 8, 16)` — the m16n8k16 Ada/Ampere FP16 atom. |
| [tensorop_gemm.py:218](../../../cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py#L218) | `MmaF16BF16Op` builds the warp-level Tensor Core atom. |
| [tensorop_gemm.py:544](../../../cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py#L544) | `LdMatrix8x8x16bOp` — CuTeDSL's alias for the `ldmatrix.x4` copy atom. |
| [tensorop_gemm.py:560](../../../cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py#L560) | `make_tiled_copy_A/B(..., tiled_mma)` — ties copy layout to MMA layout so fragments align. |
| [tensorop_gemm.py:529](../../../cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py#L529) | `thr_mma = tiled_mma.get_slice(tidx)` — maps thread to its fragment slice. |

The swizzle in the shared-memory layout is what eliminates bank conflicts for
the specific stride pattern that `ldmatrix` produces when reading those addresses.
Without the matching swizzle, `LdMatrix8x8x16bOp` would read the right values
into the right registers but the shared-memory access pattern would cause
16-way conflicts, destroying bandwidth.

The Spatters Ada blog's progression from a naive kernel to near-cuBLAS performance
is the empirical evidence for why all three contracts must align:

> *"Starting naive and fixing one issue at a time: first global-load coalescing,
> then shared-memory permutation, then `ldmatrix` + swizzle alignment...
> each step brought a measurable throughput jump."*
> — [Spatters: Ada MMA matmul](https://www.spatters.ca/mma-matmul)

### Why Not Design Around WGMMA Descriptors?

#### What WGMMA Actually Is

WGMMA (`wgmma.mma_async`) is a Hopper-introduced operation executed by an entire
**warpgroup** — four consecutive warps (128 threads) acting as one asynchronous
unit. The Colfax Hopper tutorial describes the programming model:

> *"WGMMA is a 128-thread warpgroup operation. All four warps issue the same
> `wgmma.mma_async` instruction together... The operand A and B are described
> via matrix descriptors that point into shared memory."*
> — [Colfax: WGMMA on Hopper](https://research.colfax-intl.com/cutlass-tutorial-wgmma-hopper/)

A WGMMA matrix descriptor is a 64-bit packed value with:
- Base address of the operand in shared memory
- Stride between rows
- Swizzle mode (128/64/32-byte, or none)

The descriptor replaces per-lane `ldmatrix` loads: the hardware reads the
descriptor and moves data from shared memory to tensor core inputs internally,
without per-thread address generation.

#### Why SM89 Cannot Use WGMMA

The PTX ISA is explicit:

> *"`wgmma.mma_async` requires target architecture `sm_90a`."*
> — [NVIDIA PTX ISA §9.7.14](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#asynchronous-warpgroup-level-matrix-instructions)

SM89 is below that requirement line. The instruction does not exist in the
SM89 instruction set.

#### Why the Programming Models Are Incompatible, Not Just Unavailable

Even if you could compile WGMMA for SM89, the design would be wrong:

| Dimension | Ada `mma.sync` | Hopper `wgmma.mma_async` |
|---|---|---|
| Thread group | 1 warp (32 threads) | 1 warpgroup (128 threads) |
| Synchrony | Synchronous (completes before next instruction) | Async (must `wgmma.wait_group` before reading C) |
| Operand source | Registers (`ldmatrix` → RF → MMA) | Shared memory via descriptor (no register staging) |
| Swizzle location | Software: CuTe shared-memory layout | Hardware: encoded in descriptor fields |
| Pipeline model | Explicit `cp.async` staging + `ldmatrix` | TMA + WGMMA pipeline stages |

For SM89, the swizzle is still necessary — but it lives in your CuTe layout
choice, not in a descriptor. That is the correct SM89 abstraction boundary.
The Colfax post is still worth reading as contrast: see it as "here is what the
same concept looks like two GPU generations later," not as an implementation
template.

### Why Not Design Around Tensor Maps Or TMA?

#### What TMA Is

TMA (Tensor Memory Accelerator) is a dedicated DMA-like hardware unit introduced
on Hopper. It transfers multidimensional tiles from global memory to shared
memory (or vice versa) without involving threads: one thread initiates the
transfer by issuing `cp.async.bulk` with a `CUtensorMap` descriptor, and the
hardware completes it asynchronously.

A `CUtensorMap` encodes:
- Global tensor base pointer and per-dimension strides
- Tile dimensions for the transfer
- Interleave and swizzle modes (hardware-managed, no software layout needed)

This enables **bulk one-instruction global-to-shared transfers** with
hardware-managed bank-conflict-free layout, eliminating the explicit
per-thread vectorized load + software swizzle that SM89 requires.

#### Why SM89 Cannot Use TMA

The PTX ISA is explicit:

> *"Bulk tensor copy operations require target architecture `sm_90` or higher."*
> — [NVIDIA PTX ISA §9.7.8.5](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#data-movement-and-conversion-instructions-cp-async-bulk)

SM89 does not have the TMA hardware unit. Attempting to use `cp.async.bulk` or
`CUtensorMap` on SM89 would either fail to compile or produce invalid code.

#### What SM89 Uses Instead

The SM89 equivalent of TMA is a combination of:

```text
SM89 (cp.async per-thread staging):
  each thread: ld.global.cs.v4 (vectorized 128-bit global load)
  -> cp.async.cg.shared.global (async global-to-shared copy)
  -> __pipeline_commit() / __pipeline_wait_prior()
  -> software swizzle in the CuTe layout
  -> ldmatrix from swizzled shared memory
```

vs.

```text
Hopper (TMA):
  one thread: cp.async.bulk.tensor.2d (one instruction, any tile size)
  -> hardware swizzle (from CUtensorMap descriptor)
  -> wgmma (reads directly from shared memory via descriptor)
```

When a Modular or Blackwell article describes tensor maps, TMA descriptors, or
CUtensorMap creation, the SM89 lesson to extract is: **data movement and layout
matter**. The hardware mechanism differs, but the problem is the same: get tiles
into on-chip memory in the layout that the compute instruction expects, without
bank conflicts.

### Why Not Design Around TMEM Or Blackwell `tcgen05`?

#### What TMEM Is

TMEM (Tensor Memory) is a new on-chip memory tier introduced on Blackwell (SM100)
that sits between registers and shared memory in the hierarchy:

```text
Blackwell memory hierarchy:
  HBM (global) → L2 cache → Shared Memory (SMEM) → TMEM → Registers
```

TMEM is private to a CTA and holds Tensor Core **accumulator** values — the C/D
operands of the MMA. On SM89, the accumulator lives in registers; on Blackwell,
it lives in TMEM, which is wider and avoids register file pressure from large
accumulators.

The CUTLASS/CuTeDSL Blackwell kernels use `TmemAllocator` to reserve TMEM:

> *"tcgen05.mma operates on operands where the accumulator is in TMEM rather than
> registers, allowing much larger tile sizes without register spilling."*
> — CUTLASS TMEM documentation (sm100 target docs)

TMEM capacity is 512 KB per CTA on SM100, vs the 256 KB register file.
This is what enables the Blackwell `sm100_v2` kernel in this repo to use
256×256×16 MMA tiles with `acc_stages=2` — the accumulator is never in the RF.

#### What `tcgen05` Is

`tcgen05` is the Blackwell PTX instruction family for tensor core operations:
`tcgen05.mma.cta_group::2.kind::f32`, `tcgen05.commit`, etc. The `05` refers to
the Blackwell ISA generation (tensor core generation 5).

The PTX requirement:

> *"`tcgen05.mma` requires target architecture `sm_100a` or higher."*
> — [NVIDIA PTX ISA §9.7.14 tcgen05](https://docs.nvidia.com/cuda/parallel-thread-execution/)

#### The SM89 Equivalent

On SM89, the accumulator lives in **registers**, not TMEM. This is the design
boundary:

| Feature | SM89 Ada | SM100 Blackwell |
|---|---|---|
| Accumulator location | Register file (RF) | TMEM (dedicated accumulator memory) |
| Accumulator size limit | 256 RF registers/thread (often limits tile) | 512 KB TMEM/CTA (almost no limit) |
| MMA instruction | `mma.sync.aligned.m16n8k16` | `tcgen05.mma.cta_group` |
| Ping-pong accumulators | Not applicable (RF) | `acc_stages=2` in TMEM |
| PTX requirement | `sm_89` | `sm_100a` |

When reading the Blackwell GEMM files in this repo (`sm100_v2_double_buffered.py`,
`sm100_v3_persistent_cluster.py`), translate:

```text
Blackwell: TmemAllocator → tcgen05.mma → TMEM accumulator
SM89 equivalent: registers hold C tiles → mma.sync accumulates in-place
```

The mental model is the same: keep the compute pipe fed, minimize stalls,
maximize data reuse. The storage medium for the accumulator is what differs.

### Where The 128×128×32 CTA Tile and 128-Thread Count Came From

#### The Spatters Ada Blog (Primary Source)

The Spatters post is the primary empirical source for an Ada starting point.
After sweeping multiple tile shapes on an Ada GPU, it converges on:

> *"We found that a 128×128×32 CTA tile with 128 threads was a good starting
> point... This gives a 2×2 grid of warp-level `m16n8k16` MMA atoms (4 warps
> total, matching the 4 warp schedulers on Ada)."*
> — [Spatters: Ada MMA matmul](https://www.spatters.ca/mma-matmul)

The specific chain of reasoning in the post:
1. Ada has 4 warp schedulers per SM.
2. At least 4 warps per CTA is useful so each scheduler has one warp to issue from.
3. 4 warps × 32 threads/warp = 128 threads/CTA.
4. With `atom_layout_mnk = (2, 2, 1)`, the 4 warp slots cover a 32×16 output tile
   per atom step; tiling that 4× in M and 8× in N gives 128×128.
5. `K=32` per tile: enough to keep the tensor pipe fed across the 3-stage pipeline
   without exceeding the SM's 100 KB shared memory budget.

#### The Shared Memory Budget Check

For a 128×128×32 CTA tile with 3 pipeline stages, FP16 operands:

```text
A tile:  128 × 32 × 2 bytes = 8 KB per stage
B tile:  128 × 32 × 2 bytes = 8 KB per stage
Stages:  3
Total:   (8 + 8) × 3 = 96 KB
```

The Ada tuning guide gives 100 KB as the per-block maximum:

> *"The maximum amount of shared memory per block is 99 KB on Ada."*
> — [NVIDIA Ada Tuning Guide](https://docs.nvidia.com/cuda/ada-tuning-guide/index.html)

96 KB fits within 99 KB, with 3 KB to spare for accumulators and bookkeeping.
A 128×128×32 tile with 4 stages would require 128 KB — over budget.
This is why 3 stages is the default, not 4.

#### The Local Repo Confirmation

The local `tensorop_gemm.py` encodes the same reasoning:

| Parameter | Value | Why |
|---|---|---|
| `CTA_M, CTA_N, CTA_K` | `128, 128, 32` | Matches Spatters empirical optimum and SMEM budget |
| `atom_layout_mnk` | `(2, 2, 1)` | 2×2×1 grid of warp atoms = 4 warps = 128 threads |
| `num_stages` | `3` | 3 × 16 KB = 48 KB per operand, total 96 KB, fits in 99 KB |

> *"Starting with a `(2,2,1)` atom layout and 128×128×32 CTA tile is the
> natural first configuration when targeting Ada Tensor Cores."*
> — comment in `tensorop_gemm.py` setup block

#### The Next Step: One Variable at a Time

The 128×128×32 tile is a starting point, not a conclusion. Sweep:

| Threads/CTA | Warps/CTA | When It May Help | What Can Go Wrong |
|---:|---:|---|---|
| `128` | `4` | Clean baseline; 4 warp schedulers; leaves occupancy headroom. | May not expose enough independent work for some tile shapes. |
| `256` | `8` | More warps; better scheduling if mainloop has abundant work. | Register/SMEM pressure → fewer resident CTAs/SM. |

Then sweep tile shape with threads fixed:

| CTA tile | What it tests |
|---|---|
| `128×128×32` | Square, maximizes reuse, hardest on SMEM budget |
| `128×64×32` | Rectangular — N-dimension halved, useful for non-square GEMMs |
| `64×128×32` | Rectangular — M-dimension halved |
| `64×64×32` | Small tile: more occupancy, less reuse, easier to pipeline |

```text
Procedure:
  hold MMA atom fixed at m16n8k16
  hold correctness fixed
  sweep CTA tile and atom_layout_mnk
  measure tensor pipe utilization, stalls, occupancy, bank conflicts
  change one parameter at a time
```

### Where To Go After The First SM89 Baseline

Start with the local repo's shape because it is already coherent:

```text
CTA tile:        128x128x32
MMA atom:        16x8x16
atom_layout_mnk: 2,2,1
threads/CTA:     128
stages:          3
```

Then sweep one independent variable at a time:

| Next Experiment | Why |
|---|---|
| `128x64x32` and `64x128x32` CTA tiles | Tests whether asymmetric transformer GEMMs prefer reuse along M or N. |
| `64x64x32` | Reduces shared-memory/register pressure and makes debugging easier. |
| `256` threads with `4,2,1` or similar atom layouts if the code supports it | Tests whether more warp-level MMA atoms per CTA improve issue rate. |
| 3 vs 4 pipeline stages | Tests latency hiding against shared-memory footprint. |
| A/B major-mode combinations | Tests coalescing and whether `ldmatrix.trans` is needed. |

## SM89 Strategy Summary

| Topic | SM89 Recommendation |
|---|---|
| Tensor Core instruction family | Use `mma.sync`, not `wgmma.mma_async` and not `tcgen05`. |
| Common FP16/BF16 MMA atom | Start around `m16n8k16` for FP16/BF16-style Tensor Core work. |
| Common TF32 MMA atom | For TF32 paths, study `m16n8k8`. |
| CTA thread count | Start with `128` or `256` threads per CTA. |
| Warps per CTA | `128` threads = 4 warps; `256` threads = 8 warps. |
| Resident CTA goal | Aim for roughly 2-4 resident CTAs/SM if shared memory and registers allow. |
| Shared memory strategy | Use software-controlled shared-memory layouts/swizzles through CuTe/CUTLASS. |
| Data movement | Use coalesced global loads and, where appropriate, `cp.async`-style staging. |
| Register loading | Use `ldmatrix`-compatible layouts to feed `mma.sync`. |
| What to avoid | Do not design around TMA tensor maps, `CUtensorMap`, WGMMA descriptors, TMEM, or Blackwell `tcgen05`. |

## Matrix Size And Tile-Shape Notes

The MMA instruction shape is the small matrix operation performed by one warp.
For example:

```text
mma.sync.aligned.m16n8k16...
```

means one warp cooperatively computes an MMA tile shaped:

```text
M tile: 16
N tile: 8
K tile: 16
```

A practical GEMM kernel does not stop at one MMA instruction. It builds a CTA
tile by repeating many warp-level MMA operations.

Useful CTA tile shapes to experiment with on SM89:

| CTA tile shape | Why Try It |
|---|---|
| `64x64x32` | Smaller tile, lower shared-memory pressure, useful for learning and smaller matrices. |
| `128x64x32` | Common balanced starting point; more reuse along M. |
| `64x128x32` | Common balanced starting point; more reuse along N. |
| `128x128x32` | Higher reuse, more shared memory/register pressure; often needs careful occupancy tuning. |

For LLM components:

| LLM Component | SM89 MMA Relevance |
|---|---|
| Linear layer / MLP projection | Direct GEMM. Tensor Core tiling is the main performance path. |
| Attention score `QK^T` | Direct GEMM with transpose/layout concerns. |
| Attention output `PV` | Direct GEMM after softmax. |
| Softmax | Not MMA. This is reduction/exponentiation/normalization. |
| Sigmoid/SiLU/GELU | Not MMA. These are elementwise activation kernels. |
| LayerNorm/RMSNorm | Not MMA. These are reductions and elementwise scale operations. |

## How To Validate Performance

Use Nsight Compute rather than guessing:

```bash
ncu --set full python naive_gemm_pytorch.py --mnk 4096,4096,4096
```

Look for:

```text
Tensor pipe utilization
mma instruction count
achieved occupancy
eligible warps per scheduler
DRAM throughput
shared-memory bank conflicts
L2 hit rate
```

If Tensor pipe utilization is low, the kernel is probably not issuing enough
MMA work or is starving on data movement.

If shared-memory bank conflicts are high, revisit the shared-memory layout and
the `ldmatrix` access pattern.

If occupancy is too low, inspect registers/thread and shared memory/block.

## Repo Implementation To Study First

For your `sm_89` GPU, the closest in-repo implementation is:

```text
cutedsl_examples/cute/ampere/kernel/dense_gemm/tensorop_gemm.py
```

Even though the folder says `ampere`, this is the right family of ideas for
Ada because Ada keeps the warp-level `mma.sync` style rather than Hopper
`wgmma` or Blackwell `tcgen05`.

Important choices in that file:

| Code choice | Value | Why It Matters |
|---|---:|---|
| CTA tile | `128x128x32` | One thread block owns a 128-by-128 output tile and reduces K in chunks of 32. |
| MMA instruction shape | `16x8x16` | The warp-level Tensor Core atom. This maps to the `m16n8k16` family. |
| Default atom layout | `2x2x1` | Four warp-level MMA atoms are arranged across M/N, so the CTA uses 4 warps. |
| Threads per CTA | `2 * 2 * 1 * 32 = 128` | A practical first Tensor Core CTA size on SM89. |
| Pipeline stages | `3` | Triple-buffered shared-memory staging for global-to-shared copies. |
| Global-to-shared copy | `cp.async` | Hides global-memory latency without Hopper TMA. |
| Shared-to-register copy | `ldmatrix` | Loads shared-memory tiles into the register layout expected by `mma.sync`. |
| Shared-memory layout | swizzled CuTe composed layout | Avoids shared-memory bank conflicts while feeding `ldmatrix`. |
| Epilogue staging | shared-memory C tile | Improves coalescing for stores back to global memory. |

Treat that file as your first serious template before writing from scratch.

## Recommended SM89 Development Ladder

Do not jump directly from a naive FP32 SGemm wrapper to a fully optimized
LLM GEMM. The clean path is:

| Step | Goal | What To Learn |
|---:|---|---|
| 1 | Run the existing FP32 `SGemm` wrapper | Host-side DLPack interop and correctness checks. |
| 2 | Run `tensorop_gemm.py` with FP16 inputs and FP32 accumulation | First Tensor Core path. |
| 3 | Read only the tile setup | Understand `cta_tiler`, `mma_inst_shape`, `atom_layout_mnk`, and `num_threads`. |
| 4 | Read global-to-shared copy setup | Understand coalesced 128-bit copies and `cp.async`. |
| 5 | Read shared-memory layout code | Understand why swizzle is software/CuTe layout on SM89. |
| 6 | Read `ldmatrix` setup | Understand how shared memory becomes MMA register fragments. |
| 7 | Read mainloop pipeline | Understand overlap: next global load while current K tile computes. |
| 8 | Sweep tile parameters | Measure, do not guess. |

## Tile Parameters To Sweep First

Start from the in-repo default:

```text
CTA tile:        128x128x32
MMA atom:        16x8x16
atom_layout_mnk: 2,2,1
threads/CTA:     128
stages:          3
```

Then sweep one thing at a time.

| Parameter | Try | What It Tests |
|---|---|---|
| CTA tile M/N | `128x128`, `128x64`, `64x128`, `64x64` | Reuse vs occupancy/register pressure. |
| CTA tile K | `32`, maybe `64` if supported cleanly | More K work per tile vs shared-memory footprint. |
| Atom layout | `2,2,1`, `4,1,1`, `1,4,1` | Warp distribution across M versus N. |
| Threads/CTA | `128`, `256` | More warps per block vs fewer resident CTAs. |
| Stages | `3`, `4` | More latency hiding vs more shared memory. |
| Layout major modes | A row/col, B row/col combinations | Coalescing and `ldmatrix` compatibility. |

For each sweep, record:

```text
M,N,K
CTA tile
atom layout
threads/CTA
stages
runtime
TFLOP/s
achieved occupancy
tensor pipe utilization
DRAM throughput
shared-memory bank conflicts
```

## Practical Matrix Sizes For LLM Work

For learning, choose sizes that are large enough to amortize overhead:

| Size | Why Use It |
|---|---|
| `512x512x512` | Good smoke test. Fast compile/run loop. |
| `1024x1024x1024` | Better first performance signal. |
| `2048x2048x2048` | Large enough for more stable profiling. |
| `4096x4096x4096` | Better for Nsight Compute performance counters. |

For transformer-shaped GEMMs, also test non-square cases:

| Transformer pattern | Example shape | Meaning |
|---|---|---|
| MLP up projection | `M x K` by `K x 4K` | Expands hidden dimension. |
| MLP down projection | `M x 4K` by `4K x K` | Projects back to hidden dimension. |
| Q/K/V projection | `tokens x hidden` by `hidden x hidden` | Produces attention inputs. |
| Attention score | `M x d` by `N x d` as `QK^T` | Uses transpose/layout handling. |
| Attention output | `M x N` by `N x d` | Weighted sum of values. |

Concrete starting shapes:

```text
tokens x hidden: 1024 x 1024
MLP up:          1024 x 1024 x 4096
MLP down:        1024 x 4096 x 1024
QK^T:            1024 x 128 x 1024
PV:              1024 x 1024 x 128
```

## What Good Looks Like

For an SM89 Tensor Core GEMM, a good kernel tends to have:

| Signal | What You Want |
|---|---|
| Tensor pipe utilization | High; the Tensor Core pipe should be busy. |
| Eligible warps per scheduler | Enough warps ready to issue, not constantly stalled. |
| DRAM throughput | Not the only high number; if DRAM is saturated but tensor pipe is low, you are data-starved. |
| Shared-memory bank conflicts | Low. High conflicts mean layout or `ldmatrix` access is wrong. |
| Occupancy | Enough to hide latency, but not necessarily maximum. |
| Register spilling | None or minimal. Spills usually destroy GEMM performance. |

Do not optimize for occupancy alone. A lower-occupancy kernel with better data
reuse and fewer bank conflicts can beat a higher-occupancy kernel.

## Common SM89 Failure Modes

| Failure | Symptom | Likely Cause |
|---|---|---|
| Tensor pipe underused | Low Tensor utilization in NCU | Not enough MMA work, bad pipelining, or data starvation. |
| Shared-memory conflicts | High bank conflict counters | Shared layout does not match `ldmatrix` access pattern. |
| Low occupancy | Few resident CTAs | Too much shared memory/block or too many registers/thread. |
| Bad global bandwidth | Low DRAM/L2 efficiency | Non-coalesced loads, bad major mode, or no vectorized copy. |
| Correct but slow | Passes checks, poor TFLOP/s | Usually no pipelining or poor tile shape. |
| Works for square only | Fails odd shapes | Missing predication or residue handling. |

## SM89 Rule Of Thumb

Use this as the first serious Tensor Core GEMM target:

```text
dtype:          FP16 or BF16 input
accumulator:    FP32
MMA atom:       m16n8k16
CTA tile:       128x128x32
threads/CTA:    128
atom layout:    2,2,1
stages:         3
data path:      GMEM -> cp.async -> SMEM swizzle -> ldmatrix -> mma.sync
```

Then profile before changing anything.
