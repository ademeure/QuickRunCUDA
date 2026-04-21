#!/bin/bash
# V8 J2: auto-clock for energy minimum
# For FFMA-bound AND DRAM-bound kernels, sweep GPU clock and measure:
#   - runtime (ms)
#   - avg power (W) during run — from nvidia-smi sampling
#   - derived: energy (J) = power × time; energy/op = energy / total_ops
# Reports energy optimum per workload.
#
# Usage: ./tests/bench_v8_j2_energy_sweep.sh
# Requires: sudo for nvidia-smi -lgc
set -e
cd "$(dirname "$0")/.."

echo "=== V8 J2: energy/op clock sweep ==="
echo "Note: requires sudo for nvidia-smi -lgc ... clock lock"
echo

# Build QuickRunCUDA if not built
if [ ! -f QuickRunCUDA ]; then make -s; fi

# Kill leftover processes
pkill -9 QuickRunCUDA 2>/dev/null || true
sleep 3

# Clocks to sweep (MHz)
CLOCKS=(1005 1500 1920 2032)

measure() {
    local kernel=$1
    local label=$2
    local ops_per_iter=$3
    local unit=$4  # "FLOP" or "B"
    local iters_outer=$5
    local threads=$6
    local blocks=$7

    # Launch kernel in background
    ./QuickRunCUDA -f "$kernel" -t "$threads" -b "$blocks" -0 "$iters_outer" -T 1 > /tmp/j2_out 2>&1 &
    local PID=$!

    # Sample power during run (over ~1.5 sec window)
    local pwr_samples=""
    sleep 0.3  # let kernel start
    for s in 1 2 3 4 5; do
        pwr=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
        pwr_samples="$pwr_samples $pwr"
        sleep 0.2
    done

    wait $PID
    local time_ms=$(grep "elapsed" /tmp/j2_out 2>/dev/null | awk '{print $(NF-1)}' | head -1)
    if [ -z "$time_ms" ]; then
        time_ms=$(grep -oP '\d+\.\d+ ms' /tmp/j2_out | head -1 | awk '{print $1}')
    fi

    # Average power
    local pwr_avg=$(echo $pwr_samples | tr ' ' '\n' | awk '{s+=$1; n++} END {if(n>0) printf "%.1f", s/n; else print "0"}')
    local pwr_max=$(echo $pwr_samples | tr ' ' '\n' | awk '{if($1>m) m=$1} END {printf "%.1f", m}')

    if [ -z "$time_ms" ] || [ "$time_ms" == "0" ]; then
        echo "  $label: MEASUREMENT FAILED (no time)"
        return
    fi

    local time_s=$(echo "$time_ms / 1000" | bc -l)
    local total_ops=$(echo "$ops_per_iter * $iters_outer * $threads * $blocks" | bc -l)
    local energy_J=$(echo "$pwr_avg * $time_s" | bc -l)
    local energy_per_op=$(echo "$energy_J / $total_ops" | bc -l)

    printf "  %-12s time=%6.1f ms  pwr_avg=%5.1f W  (pk %5.1f)  E=%6.2f J   E/op=%.3e J/%s\n" \
        "$label" "$time_ms" "$pwr_avg" "$pwr_max" "$energy_J" "$energy_per_op" "$unit"
}

# === 1. Build FFMA kernel ===
echo "Compiling bench kernels..."
# Use existing V6 C1 FFMA (compute-bound) and V6 C2 DRAM
FFMA_KERNEL=tests/bench_v6_c1_ffma_energy.cu
DRAM_KERNEL=tests/bench_v6_c2_dram_energy.cu

# FFMA kernel: ITERS_INNER=65536, 8 chains, 2 FLOP/FMA
# Total FLOPs = iters_outer * 65536 * 8 * 2
FFMA_OPS_PER_ITER=$((65536 * 8 * 2))  # per thread per outer iter
# DRAM kernel: reads N=16384 float4 per thread per outer iter = 16384 * 16 bytes = 256 KB
DRAM_OPS_PER_ITER=$((16384 * 16))  # bytes per thread per outer iter

# Threads × blocks to use:
# For FFMA: 256 × 148 = full occupancy
# For DRAM: 256 × 132 = pull ~32 GB/s per thread is silly, use 256 × 148

for clk in "${CLOCKS[@]}"; do
    echo
    echo "--- Clock target: $clk MHz ---"
    sudo nvidia-smi -lgc $clk -i 0 > /dev/null 2>&1 || echo "  (lock may need sudo or different syntax)"
    sleep 2
    actual=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
    echo "  Actual clock: $actual MHz"

    measure "$FFMA_KERNEL" "FFMA_2s"  $FFMA_OPS_PER_ITER "FLOP" 100 256 148
    measure "$DRAM_KERNEL" "DRAM_2s"  $DRAM_OPS_PER_ITER "B"    10  256 148
done

echo
echo "--- Unlock clocks ---"
sudo nvidia-smi -rgc -i 0 > /dev/null 2>&1 || true
sleep 1

echo
echo "=== Interpretation ==="
echo "E/op lowest at which clock?"
echo "  FFMA (compute-bound): boost clock usually WINS (per-op energy drops w/ clock due to V² DVS)"
echo "  DRAM (mem-bound): lower clock WINS (time bounded by BW, lower clock = less cores spinning)"
