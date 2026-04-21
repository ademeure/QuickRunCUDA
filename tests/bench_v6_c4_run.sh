#!/bin/bash
# V6 C4: SINGLE-WARP FFMA energy sweep
# Test: does low occupancy change energy minimum?
set -e

CLOCKS=(510 800 1005 1500 1992)
ITERS_INNER=65536
FFMA_PER_INNER=8
N_THREADS=32
ITERS_OUTER=2000

echo "RIGOR — V6 C4 SINGLE-WARP FFMA energy sweep"
printf "%-7s %-9s %-8s %-9s %-12s %-12s\n" "CLK" "Pavg(W)" "Pact(W)" "time(ms)" "GFLOPS" "pJ/FFMA"
echo "---------------------------------------------------------------------"

for CLK in "${CLOCKS[@]}"; do
    sudo nvidia-smi -i 0 -lgc $CLK,$CLK > /dev/null 2>&1
    sleep 1
    BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
    sleep 0.2

    timeout 60 ./QuickRunCUDA -f tests/bench_v6_c4_singlewarp_energy.cu -t $N_THREADS -b 1 \
        -0 $ITERS_OUTER -A 1024 -B 1024 -C 1024 -T 3 > /tmp/c4_kern.out 2>&1 &
    KERN_PID=$!
    sleep 1.0

    PSUM=0
    for i in 1 2 3 4 5 6; do
        PSAMP=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
        PSUM=$(echo "$PSUM + $PSAMP" | bc)
        sleep 0.2
    done
    AVG_P=$(echo "scale=2; $PSUM / 6" | bc)
    ACT_P=$(echo "scale=2; $AVG_P - $BASE_W" | bc)

    wait $KERN_PID 2>/dev/null
    KTIME_MS=$(grep -oE "[0-9.]+ ms$" /tmp/c4_kern.out | head -1 | awk '{print $1}')

    if [ -n "$KTIME_MS" ] && [ "$KTIME_MS" != "0" ]; then
        TOTAL_FFMA=$(python3 -c "print($N_THREADS * $ITERS_INNER * $FFMA_PER_INNER * $ITERS_OUTER)")
        GFLOPS=$(python3 -c "print(f'{$TOTAL_FFMA * 2 / ($KTIME_MS/1000) / 1e9:.2f}')")
        PJ_PER_FFMA=$(python3 -c "print(f'{$ACT_P * ($KTIME_MS/1000) / $TOTAL_FFMA * 1e12:.0f}')")
        printf "%-7s %-9s %-8s %-9s %-12s %-12s\n" "$CLK" "$AVG_P" "$ACT_P" "$KTIME_MS" "$GFLOPS" "$PJ_PER_FFMA"
    fi
done

sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
echo "(restored)"
