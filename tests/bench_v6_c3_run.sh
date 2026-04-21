#!/bin/bash
# V6 C3: mixed FFMA+DRAM min-energy sweep
set -e

CLOCKS=(510 800 1005 1500 1992)
NSMS=148
THREADS=256
ITERS_INNER=32768
ITERS_OUTER=200
FFMA_PER_OUTER=$((NSMS * THREADS * ITERS_INNER * 4))    # 4 FFMA per inner iter
LDG_PER_OUTER=$((NSMS * THREADS * ITERS_INNER * 16))    # 16 B per LDG

echo "RIGOR — V6 C3 mixed FFMA+DRAM min-energy sweep"
printf "%-7s %-9s %-8s %-9s %-12s %-12s\n" "CLK" "Pavg(W)" "Pact(W)" "time(ms)" "TFLOPS" "BW(GB/s)"
echo "---------------------------------------------------------------------"

for CLK in "${CLOCKS[@]}"; do
    sudo nvidia-smi -i 0 -lgc $CLK,$CLK > /dev/null 2>&1
    sleep 1
    BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
    sleep 0.2

    timeout 60 ./QuickRunCUDA -f tests/bench_v6_c3_mixed_energy.cu -t $THREADS -p \
        -0 $ITERS_OUTER -A 16777216 -B 1024 -C 1048576 -T 3 > /tmp/c3_kern.out 2>&1 &
    KERN_PID=$!
    sleep 1.2

    PSUM=0
    for i in 1 2 3 4 5 6; do
        PSAMP=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
        PSUM=$(echo "$PSUM + $PSAMP" | bc)
        sleep 0.2
    done
    AVG_P=$(echo "scale=2; $PSUM / 6" | bc)
    ACT_P=$(echo "scale=2; $AVG_P - $BASE_W" | bc)

    wait $KERN_PID 2>/dev/null
    KTIME_MS=$(grep -oE "[0-9.]+ ms$" /tmp/c3_kern.out | head -1 | awk '{print $1}')

    if [ -n "$KTIME_MS" ] && [ "$KTIME_MS" != "0" ]; then
        TFLOPS=$(python3 -c "print(f'{$FFMA_PER_OUTER * $ITERS_OUTER * 2 / ($KTIME_MS/1000) / 1e12:.2f}')")
        BW=$(python3 -c "print(f'{$LDG_PER_OUTER * $ITERS_OUTER / ($KTIME_MS/1000) / 1e9:.0f}')")
        printf "%-7s %-9s %-8s %-9s %-12s %-12s\n" "$CLK" "$AVG_P" "$ACT_P" "$KTIME_MS" "$TFLOPS" "$BW"
    else
        echo "ERROR at clock $CLK"
    fi
done

sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
echo "(restored to 1500 MHz)"
