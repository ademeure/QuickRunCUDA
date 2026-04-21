#!/bin/bash
# V6 C1: rigorous FFMA min-energy sweep
# For each clock: lock clock, run kernel for ~5 sec, sample power, compute pJ/FFMA
set -e

CLOCKS=(510 800 1005 1200 1400 1500 1700 1992)
NSMS=148
THREADS=256
ITERS_INNER=65536
FLOPS_PER_INNER=8           # 8 FFMAs (anti-DCE chain)
FLOPS_PER_FMA=2             # mul + add per FFMA
THREADS_TOTAL=$((NSMS * THREADS))

echo "RIGOR PROTOCOL — V6 C1 FFMA min-energy clock sweep"
echo ""
echo "Theoretical FFMAs per outer iter:"
echo "  $NSMS SMs × $THREADS thr × $ITERS_INNER × $FLOPS_PER_INNER = $((NSMS * THREADS * ITERS_INNER * FLOPS_PER_INNER))"
echo ""

ITERS_OUTER=500

# Header
printf "%-7s %-9s %-8s %-9s %-12s %-13s\n" "CLK" "Pavg(W)" "Pact(W)" "time(ms)" "TFLOP/s" "pJ/FFMA"
echo "----------------------------------------------------------------------"

for CLK in "${CLOCKS[@]}"; do
    sudo nvidia-smi -i 0 -lgc $CLK,$CLK > /dev/null 2>&1
    sleep 1

    # Get baseline (idle) power for delta
    BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
    sleep 0.2

    # Run kernel in background, capture stderr/stdout
    timeout 60 ./QuickRunCUDA -f tests/bench_v6_c1_ffma_energy.cu -t $THREADS -p \
        -0 $ITERS_OUTER -A 1024 -B 1024 -C 1048576 -T 3 > /tmp/c1_kern.out 2>&1 &
    KERN_PID=$!
    sleep 1.5  # warmup

    # Sample power 8 times during kernel run
    PSUM=0; CSUM=0
    for i in 1 2 3 4 5 6 7 8; do
        SAMPLE=$(nvidia-smi --query-gpu=power.draw,clocks.current.graphics --format=csv,noheader,nounits -i 0)
        PSAMP=$(echo $SAMPLE | awk -F',' '{print $1}')
        CSAMP=$(echo $SAMPLE | awk -F',' '{print $2}')
        PSUM=$(echo "$PSUM + $PSAMP" | bc)
        CSUM=$(echo "$CSUM + $CSAMP" | bc)
        sleep 0.2
    done
    AVG_P=$(echo "scale=2; $PSUM / 8" | bc)
    AVG_C=$(echo "scale=0; $CSUM / 8" | bc)
    ACT_P=$(echo "scale=2; $AVG_P - $BASE_W" | bc)

    wait $KERN_PID 2>/dev/null

    # Get kernel time (from -T 5 timing)
    KTIME_MS=$(grep -oE "[0-9.]+ ms$" /tmp/c1_kern.out | head -1 | awk '{print $1}')

    if [ -n "$KTIME_MS" ] && [ "$KTIME_MS" != "0" ]; then
        TOTAL_FFMA=$(python3 -c "print($NSMS * $THREADS * $ITERS_INNER * $FLOPS_PER_INNER * $ITERS_OUTER)")
        TOTAL_FLOPS=$(python3 -c "print($TOTAL_FFMA * $FLOPS_PER_FMA)")
        TFLOPS=$(python3 -c "print(f'{$TOTAL_FLOPS / ($KTIME_MS / 1000) / 1e12:.2f}')")
        # pJ/FFMA = active_W × time_s / total_ffmas × 1e12
        PJ_PER_FFMA=$(python3 -c "print(f'{$ACT_P * ($KTIME_MS/1000) / $TOTAL_FFMA * 1e12:.2f}')")
        printf "%-7s %-9s %-8s %-9s %-12s %-13s\n" "$AVG_C" "$AVG_P" "$ACT_P" "$KTIME_MS" "$TFLOPS" "$PJ_PER_FFMA"
    else
        echo "ERROR: no kernel time captured at clock $CLK"
        cat /tmp/c1_kern.out | tail -5
    fi
done

# Restore default
sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
echo "(restored to 1500 MHz)"
