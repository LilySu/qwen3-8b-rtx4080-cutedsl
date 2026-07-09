# profiles/

NCU binary reports and the written findings they produced.

## Relationship to bench/results/

`bench/results/` answers **how fast**.  
`profiles/` answers **why**.

Run `bench/runner.py` after every iteration.  
Create a profile only when a bench result needs diagnosis — worse than expected,
or before committing to a new optimization strategy.

## Rules

1. **Profile filename = kernel_version_regime** — must match the `kernel`, `version`,
   and `regime` fields in the corresponding `bench/results/run_*.json` record.
   Example: `rope_v1_cute_naive_decode`

2. **Every `.ncu-rep` requires a paired `.md`** before the iteration is considered
   complete. The `.ncu-rep` is gitignored; the `.md` is committed.

3. **The `.md` must contain three things:**
   - The bench row from `ITERATIONS.md` (copy the numbers)
   - The NCU finding: bottleneck type, achieved occupancy, limiting factor
   - The next hypothesis (what you'll change in the next version)

## When to profile

Profile when:
- `pct_peak_bw` is below 60% for a decode kernel (should eventually reach ~85–90%)
- `pct_peak_flops` is below 50% for a prefill kernel and you don't know why
- A version is slower than its predecessor

Skip profiling when:
- The bench number is expected (e.g., v0 PyTorch at ~30% BW is normal)
- You already know the bottleneck from the arithmetic (e.g., decode GEMVs are always BW-bound)

## NCU collection

```bash
bash ncu_collect.sh <kernel> <version> <regime>
# example:
bash ncu_collect.sh rope v1_cute_naive decode
```

Open the `.ncu-rep` on Windows with Nsight Compute UI.  
Prerequisite: NVIDIA Control Panel → "Allow access to GPU performance counters to all users".

## Template for the paired .md

```markdown
# <kernel> · <version> · <regime>

## Bench result
(paste the row from ITERATIONS.md)

## NCU findings
- Bottleneck: [memory-bound / compute-bound / launch overhead]
- Achieved occupancy: X warps / 48 max
- Limiting factor: (e.g., "bank conflicts in SMEM load", "L2 miss rate 80%")
- Achieved BW: X GB/s of 432 GB/s theoretical

## Next hypothesis
(what to change in the next version and why)
```
