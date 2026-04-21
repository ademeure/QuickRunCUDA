#!/bin/bash
# V5 A2: measure power for each waiting mode
set -e
ITERS=8000000  # 8 sec hold time per block (long enough to sample during wait)

for MODE in 0 1 2; do
    # Launch kernel in background
    timeout 60 ./QuickRunCUDA -f tests/bench_v5_a2_mbarrier_power.cu -t 256 -p \
        -H "#define MODE $MODE" -0 $ITERS -A 1024 -B 1024 -C 1048576 > /tmp/a2_kern.out 2>&1 &
    KERN_PID=$!

    sleep 1.0  # let it warm up

    # Sample power 8x at 200ms intervals (during the spin period)
    POWER_SAMPLES=$(for i in {1..8}; do
        nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0
        sleep 0.2
    done)
    AVG_POWER=$(echo "$POWER_SAMPLES" | awk '{sum+=$1; n++} END {printf "%.1f", sum/n}')

    wait $KERN_PID 2>/dev/null

    case $MODE in
        0) NAME="busy_spin" ;;
        1) NAME="mbarrier_try_wait" ;;
        2) NAME="nanosleep_100" ;;
    esac
    echo "MODE=$MODE ($NAME): avg_power=$AVG_POWER W"
done
