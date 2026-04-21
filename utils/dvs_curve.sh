#!/bin/bash
# V10: DVS voltage curve — FFMA workload power across clock range
set -e
cd "$(dirname "$0")/.."

if [ ! -f QuickRunCUDA ]; then make -s; fi

pkill -9 QuickRunCUDA 2>/dev/null || true
sleep 3

# Test clocks (MHz) — sample broadly
CLOCKS=(510 800 1005 1200 1500 1700 1920 2032)

echo "=== DVS power curve for sustained FFMA ==="
echo "Clock  Time(ms)  Idle(W)  FFMA(W)  Delta(W)  TFLOPS  GFLOPS/W"

# Use V6 C1 FFMA kernel (long chain)
KERNEL=tests/bench_v6_c1_ffma_energy.cu

for clk in "${CLOCKS[@]}"; do
    sudo nvidia-smi -lgc $clk -i 0 > /dev/null 2>&1 || true
    sleep 2

    # Sample idle
    IDLE_SUM=0; N=0
    for i in 1 2 3 4 5; do
        p=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
        IDLE_SUM=$(echo "$IDLE_SUM + $p" | bc)
        N=$((N + 1))
        sleep 0.2
    done
    IDLE=$(echo "scale=1; $IDLE_SUM / $N" | bc)

    # Run kernel and sample power during
    ./QuickRunCUDA -f $KERNEL -t 256 -b 148 -0 3000 -T 1 > /tmp/dvs_out 2>&1 &
    PID=$!
    sleep 0.5
    PWR_SUM=0; NP=0
    for i in 1 2 3 4 5 6 7 8; do
        p=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
        PWR_SUM=$(echo "$PWR_SUM + $p" | bc)
        NP=$((NP + 1))
        sleep 0.3
    done
    wait $PID
    PWR=$(echo "scale=1; $PWR_SUM / $NP" | bc)
    TIME=$(grep -oP '\d+\.\d+' /tmp/dvs_out | head -1)
    DELTA=$(echo "scale=1; $PWR - $IDLE" | bc)

    # TFLOPS: 3000 × 65536 × 8 × 2 × 256 × 148 / time
    OPS=$(echo "3000 * 65536 * 8 * 2 * 256 * 148" | bc)
    TFLOPS=$(echo "scale=2; $OPS / $TIME / 1000 / 1000 / 1000" | bc)
    EFF=$(echo "scale=2; $TFLOPS * 1000 / $PWR" | bc)

    printf "%5d  %8s  %7s  %7s  %7s  %6s  %7s\n" "$clk" "$TIME" "$IDLE" "$PWR" "$DELTA" "$TFLOPS" "$EFF"
done

sudo nvidia-smi -rgc -i 0 > /dev/null 2>&1 || true
echo "(clock unlocked)"
