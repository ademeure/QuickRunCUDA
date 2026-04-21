#!/bin/bash
# V7 K4: scale blocks to test SM power-gating
sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
sleep 1

for B in 1 4 16 37 74 148; do
    BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
    timeout 30 ./QuickRunCUDA -f tests/bench_v7_k4_sm_sleep.cu -t 256 -b $B -0 5000 -A 1024 -B 1024 -C 1048576 > /tmp/k4.out 2>&1 &
    KERN_PID=$!
    sleep 1.5

    PSUM=0
    for i in 1 2 3 4 5 6; do
        PSAMP=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
        PSUM=$(echo "$PSUM + $PSAMP" | bc)
        sleep 0.2
    done
    AVG_P=$(echo "scale=2; $PSUM / 6" | bc)
    ACT_P=$(echo "scale=2; $AVG_P - $BASE_W" | bc)

    wait $KERN_PID 2>/dev/null
    printf "Blocks=%-4d Pavg=%6sW Pact=%6sW (per-block ≈ %sW)\n" "$B" "$AVG_P" "$ACT_P" "$(echo "scale=2; $ACT_P / $B" | bc)"
done
