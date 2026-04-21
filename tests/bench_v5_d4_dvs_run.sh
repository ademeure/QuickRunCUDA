#!/bin/bash
# V5 D4: sweep clock and measure FFMA power
set -e
CLOCKS=(510 1005 1500 1700 1992)

for CLK in "${CLOCKS[@]}"; do
    sudo nvidia-smi -i 0 -lgc $CLK,$CLK > /dev/null 2>&1
    sleep 1

    # Run kernel in background while sampling power
    timeout 30 ./QuickRunCUDA -f tests/bench_v5_d4_dvs.cu -t 256 -p -0 5000 -A 1024 -B 1024 -C 1048576 > /tmp/d4_kern.out 2>&1 &
    KERN_PID=$!
    sleep 2  # wait for warmup

    # Sample power 10x at 100ms intervals
    POWER_SAMPLES=$(for i in {1..10}; do nvidia-smi --query-gpu=power.draw,clocks.current.graphics --format=csv,noheader,nounits -i 0; sleep 0.1; done)

    wait $KERN_PID

    AVG_POWER=$(echo "$POWER_SAMPLES" | awk -F',' '{sum+=$1; n++} END {printf "%.1f", sum/n}')
    AVG_CLK=$(echo "$POWER_SAMPLES" | awk -F',' '{sum+=$2; n++} END {printf "%.0f", sum/n}')
    KERN_TIME=$(grep -oE "[0-9.]+ ms" /tmp/d4_kern.out | head -1)

    echo "TARGET_CLK=$CLK MHz | actual_avg=$AVG_CLK MHz | power=$AVG_POWER W | kern_time=$KERN_TIME"
done

# Restore default clock
sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
echo "(restored to 1500 MHz)"
