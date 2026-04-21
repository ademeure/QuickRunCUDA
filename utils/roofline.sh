#!/bin/bash
# B300 Roofline plotter
# Usage: ./utils/roofline.sh <kernel_binary> [args]
# Computes FLOP rate vs arithmetic intensity using ncu metrics
# Plots ASCII roofline with computed bound

if [ $# -lt 1 ]; then
    echo "Usage: $0 <binary> [args]" >&2
    echo "  Outputs: AI (FLOP/byte), GFLOPS, % roofline" >&2
    exit 1
fi

# B300 peaks (sm_103a, 1500 MHz locked)
HBM_PEAK_GBPS=7500    # ~7.5 TB/s HBM3E peak
FFMA_PEAK_TFLOPS=57   # 148 SMs × 128 cores × 1.5 GHz × 2 = 57 TFLOPS
TENSOR_PEAK_TFLOPS=540 # FP16 tensor peak

# Run ncu to capture FLOPs and bytes
OUTPUT=$(sudo -n /usr/local/cuda/bin/ncu --metrics \
    smsp__inst_executed_pipe_fma.sum,\
sm__pipe_tensor_cycles_active.sum,\
dram__bytes_read.sum,\
dram__bytes_write.sum,\
sm__cycles_elapsed.sum \
    --csv "$@" 2>&1 | grep -v "^==")

extract_value() {
    echo "$OUTPUT" | grep "$1" | tail -1 | awk -F'","' '{v=$NF; gsub("[\",]","",v); print v}'
}

FFMA_INST=$(extract_value "smsp__inst_executed_pipe_fma.sum")
TENSOR_CY=$(extract_value "sm__pipe_tensor_cycles_active.sum")
DRAM_RD=$(extract_value "dram__bytes_read.sum")
DRAM_WR=$(extract_value "dram__bytes_write.sum")
ELAPSED=$(extract_value "sm__cycles_elapsed.sum")

# Compute FLOPs (warp inst × 32 threads × 2 FLOPs per FFMA)
FFMA_FLOPS=$((FFMA_INST * 32 * 2))
DRAM_BYTES=$((DRAM_RD + DRAM_WR))

# Arithmetic intensity
if [ "$DRAM_BYTES" -gt 0 ]; then
    AI=$(echo "scale=2; $FFMA_FLOPS / $DRAM_BYTES" | bc)
else
    AI="inf"
fi

# Time at 1500 MHz: cycles / 1.5 GHz / 148 SMs = seconds
TIME_S=$(echo "scale=6; $ELAPSED / 148 / 1500000000" | bc)
GFLOPS=$(echo "scale=2; $FFMA_FLOPS / $TIME_S / 1e9" | bc 2>/dev/null)

# Roofline knee: AI_knee = peak_FLOPS / peak_BW
AI_KNEE=$(echo "scale=2; $FFMA_PEAK_TFLOPS * 1000 / $HBM_PEAK_GBPS" | bc)

echo ""
echo "=================================================="
echo " B300 Roofline (assuming 1500 MHz locked)"
echo "=================================================="
printf "%-20s %s\n" "FFMA warp insts:" "$FFMA_INST"
printf "%-20s %.2e\n" "Total FFMA FLOPs:" "$FFMA_FLOPS"
printf "%-20s %.2e B\n" "DRAM bytes:" "$DRAM_BYTES"
printf "%-20s %s FLOP/byte\n" "Arithmetic intensity:" "$AI"
printf "%-20s %s\n" "Measured GFLOPS:" "$GFLOPS"
printf "%-20s %s FLOP/byte (FFMA-bound vs HBM-bound)\n" "Roofline knee:" "$AI_KNEE"
echo ""
if [ -n "$AI" ] && [ "$AI" != "inf" ]; then
    if (( $(echo "$AI < $AI_KNEE" | bc -l) )); then
        echo "→ MEMORY-BOUND (AI=$AI < knee=$AI_KNEE)"
        ROOF_GFLOPS=$(echo "scale=2; $AI * $HBM_PEAK_GBPS" | bc)
        echo "  Roofline limit: $ROOF_GFLOPS GFLOPS"
    else
        echo "→ COMPUTE-BOUND (AI=$AI >= knee=$AI_KNEE)"
        echo "  Roofline limit: $((FFMA_PEAK_TFLOPS * 1000)) GFLOPS (peak FFMA)"
    fi
fi
echo "=================================================="
