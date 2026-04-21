#!/bin/bash
# V8 L2: block-size auto-bisect
# Usage: ./utils/block_bisect.sh <kernel.cu> [other QuickRunCUDA args...]
# Sweeps block sizes {32, 64, 128, 256, 512, 1024} with blocks auto-scaled
# to target 148-296 total blocks (full-occupancy range on B300).
# Reports time per config, highlights optimum.

set -e
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <kernel.cu> [additional QuickRunCUDA flags]"
    echo "  e.g.  $0 tests/bench_v6_c1_ffma_energy.cu -0 200 -A 100000000"
    exit 1
fi

KERNEL="$1"
shift
EXTRA_ARGS="$@"

if [ ! -f "$KERNEL" ]; then echo "Kernel not found: $KERNEL"; exit 1; fi
if [ ! -f ./QuickRunCUDA ]; then make -s; fi

echo "=== V8 L2: block-size auto-bisect ==="
echo "Kernel: $KERNEL"
echo "Extra args: $EXTRA_ARGS"
echo

pkill -9 QuickRunCUDA 2>/dev/null || true
sleep 2

declare -A TIMES
BLOCKS_PER_THREADS=""

# Keep total threads constant at ~37888 (148 SMs × 256 occupancy).
# Vary blocks to compensate: blocks = 37888 / threads_per_block.
TOTAL_THREADS=37888
for THREADS in 32 64 128 256 512 1024; do
    BLOCKS=$((TOTAL_THREADS / THREADS))
    [ $BLOCKS -lt 1 ] && BLOCKS=1

    T1=$(./QuickRunCUDA -f "$KERNEL" -t $THREADS -b $BLOCKS $EXTRA_ARGS -T 5 2>/dev/null | grep -oP '\d+\.\d+' | head -1)
    if [ -z "$T1" ]; then T1="0"; fi
    TIMES[$THREADS]=$T1
    printf "  t=%4d b=%4d (total=%d):  %s ms\n" "$THREADS" "$BLOCKS" "$((BLOCKS * THREADS))" "$T1"
done

echo
echo "--- Analysis ---"
BEST_T=32
BEST_TIME=99999
for t in 32 64 128 256 512 1024; do
    v="${TIMES[$t]}"
    if [ "$v" != "0" ]; then
        if awk "BEGIN{exit !($v < $BEST_TIME)}"; then
            BEST_TIME=$v
            BEST_T=$t
        fi
    fi
done

echo "Optimal block size: $BEST_T threads/block ($BEST_TIME ms)"
echo
echo "Relative performance (vs optimum):"
for t in 32 64 128 256 512 1024; do
    v="${TIMES[$t]}"
    if [ "$v" != "0" ]; then
        RATIO=$(awk "BEGIN{printf \"%.2f\", $v / $BEST_TIME}")
        printf "  t=%4d:  %s×\n" "$t" "$RATIO"
    fi
done

echo
echo "Guidance (from V8 J1):"
echo "  FFMA-bound  → 128 threads/block typically optimal (4 warps = 1/SMSP)"
echo "  MMA-bound   → 128 threads/block (tensor pipe prefers 1 warp group)"
echo "  DRAM-bound  → 256-512 (more warps for latency hiding)"
echo "  Register-heavy → lower blocks (32-64) to keep occupancy"
