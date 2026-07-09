#!/usr/bin/env bash
# Collect an Nsight Compute profile for a single kernel version.
#
# Usage (from inside the container):
#   bash ncu_collect.sh rope v0_pytorch decode
#   bash ncu_collect.sh gemm v1_cutedsl prefill
#
# Kernel short names (same as bench/runner.py):
#   rmsnorm   attention   gemm   rope   mlp
#
# Output: profiles/<kernel>_<version>_<regime>.ncu-rep
#
# Prerequisites (WSL2):
#   - On Windows host: NVIDIA Control Panel → "Allow access to GPU performance
#     counters to all users" (otherwise every ncu run fails with ERR_NVGPUCTRPERM)
#   - Nsight Compute UI installed on Windows for report viewing
#
# Sections collected: LaunchStats, Occupancy, MemoryWorkloadAnalysis,
#   ComputeWorkloadAnalysis — enough for roofline without a full-speed replay.

set -euo pipefail

KERNEL="${1:?Usage: ncu_collect.sh <kernel> <version> <regime>}"
VERSION="${2:?}"
REGIME="${3:-decode}"

OUT_DIR="profiles"
mkdir -p "$OUT_DIR"
OUT="$OUT_DIR/${KERNEL}_${VERSION}_${REGIME}.ncu-rep"

# Inline benchmark script — uses runner's _discover_versions so it respects
# the KERNELS map (short name → numbered directory).
BENCH_SCRIPT=$(cat <<PYEOF
import sys, torch
sys.path.insert(0, ".")
from bench.runner import _discover_versions, _decode_shapes, _prefill_shapes

shapes = _decode_shapes() if "${REGIME}" == "decode" else _prefill_shapes()
args_fn = shapes.get("${KERNEL}")
if args_fn is None:
    sys.exit(f"no shape defined for kernel '${KERNEL}'")
args = args_fn()

versions = _discover_versions("${KERNEL}")
mod = dict(versions).get("${VERSION}")
if mod is None:
    sys.exit(f"version '${VERSION}' not found for kernel '${KERNEL}'")

# Warm up, then run exactly once for ncu to capture
for _ in range(5):
    mod.benchmark(*args, warmup=5, iters=1)
torch.cuda.synchronize()
mod.benchmark(*args, warmup=0, iters=1)
PYEOF
)

echo "Profiling: kernel=${KERNEL}  version=${VERSION}  regime=${REGIME}"
echo "Output:    ${OUT}"

ncu \
  --set default \
  --section LaunchStats \
  --section Occupancy \
  --section MemoryWorkloadAnalysis \
  --section ComputeWorkloadAnalysis \
  --kernel-name-base function \
  --launch-skip 5 \
  --launch-count 1 \
  --export "${OUT}" \
  python3 -c "$BENCH_SCRIPT"

echo "Done. Open ${OUT} in Nsight Compute on Windows."
