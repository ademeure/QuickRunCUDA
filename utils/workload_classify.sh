#!/bin/bash
# B300 workload classifier — analyzes a kernel and recommends:
# - Optimal clock (min energy)
# - Optimal block size
# - Roofline class (compute vs memory bound)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <binary> [args...]"
    exit 1
fi

cd /root/github/QuickRunCUDA

# Run ncu to collect metrics
OUTPUT=$(sudo -n /usr/local/cuda/bin/ncu --metrics \
    sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,\
sm__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_elapsed,\
lts__t_sector_hit_rate.pct,\
dram__bytes_read.sum,\
smsp__inst_executed.sum,\
launch__registers_per_thread \
    "$@" 2>&1 | grep -v "^==")

get_val() {
    # Extract last whitespace-separated numeric field from metric line
    echo "$OUTPUT" | grep -F "$1" | tail -1 | awk '{
        for (i=NF; i>=1; i--) {
            v=$i; gsub(",","",v);
            if (v ~ /^[0-9.]+$/) { print v; exit }
        }
        print "0"
    }'
}

FMA=$(get_val "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed")
TENSOR=$(get_val "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed")
LSU=$(get_val "sm__pipe_lsu_cycles_active.avg.pct_of_peak_sustained_elapsed")
L2_HIT=$(get_val "lts__t_sector_hit_rate.pct")
DRAM_RD=$(get_val "dram__bytes_read.sum")
REGS=$(get_val "launch__registers_per_thread")

echo ""
echo "=============================================="
echo " B300 Workload Classifier"
echo "=============================================="
printf "%-25s %s%%\n" "FMA pipe peak util:" "$FMA"
printf "%-25s %s%%\n" "Tensor pipe peak util:" "$TENSOR"
printf "%-25s %s%%\n" "LSU pipe peak util:" "$LSU"
printf "%-25s %s%%\n" "L2 hit rate:" "$L2_HIT"
printf "%-25s %s MB\n" "DRAM read total:" "$DRAM_RD"
printf "%-25s %s\n" "Regs/thread:" "$REGS"

echo ""
echo "=============================================="
echo " Recommendations"
echo "=============================================="

FMA_BOUND=$(echo "$FMA > 50" | bc -l 2>/dev/null || echo 0)
TENSOR_BOUND=$(echo "$TENSOR > 50" | bc -l 2>/dev/null || echo 0)
LSU_BOUND=$(echo "$LSU > 30" | bc -l 2>/dev/null || echo 0)

if [ "$FMA_BOUND" = "1" ]; then
    echo "→ FMA-BOUND workload"
    echo "  Min-energy clock: 510 MHz (V6 C1: 16% savings, 5.76 pJ/FFMA)"
    echo "  Block size: 128 (V8 J1: 1 warp/SMSP saturation)"
elif [ "$TENSOR_BOUND" = "1" ]; then
    echo "→ TENSOR-BOUND workload"
    echo "  Min-energy clock: BOOST 1992 MHz (V6 C3: mixed ML 3× lower)"
    echo "  Block size: 128-256 (cuTLASS warpgroup = 4 chains = 128 thr)"
    echo "  Consider cluster CSIZE=4-8 for DSMEM (V7 D2: 2.4× faster bcast)"
elif [ "$LSU_BOUND" = "1" ]; then
    echo "→ MEMORY-BOUND workload"
    echo "  Min-energy clock: 800 MHz (V6 C2: 36% savings, 11.81 pJ/byte)"
    echo "  Block size: 256-512 (higher occupancy helps V7 G3)"
    echo "  Use prefetch.L2 + cp.async depth=16 (V7 I1: 15× speedup)"
else
    echo "→ MIXED / LOW-UTIL workload"
    echo "  Min-energy clock: BOOST 1992 MHz (V6 C3: static power amortization)"
    echo "  Consider higher block size (256+) to expose ILP"
fi

echo ""
echo "Cross-checks:"
L2_POOR=$(echo "$L2_HIT < 50" | bc -l 2>/dev/null || echo 0)
if [ "$L2_POOR" = "1" ]; then
    echo "  Low L2 hit (${L2_HIT}%) — likely random access pattern (V7 G2: 35% typical)"
    echo "  Consider: sort/tile access for sequential (V7 G2 shows 94% hit)"
fi
if [ "${REGS:-0}" -gt 128 ]; then
    echo "  High reg/thread ($REGS) — check occupancy; spills likely at >255"
fi
echo "=============================================="
