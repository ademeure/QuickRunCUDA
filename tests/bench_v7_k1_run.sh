#!/bin/bash
# V7 K1: sustained throttle test
# Unlock clock first, then run FFMA-heavy and observe boost behavior
sudo nvidia-smi -i 0 -rgc > /dev/null 2>&1
sleep 1
echo "Unlocked clock state:"
nvidia-smi --query-gpu=power.draw,clocks.current.graphics --format=csv,noheader,nounits | head -1

# Launch kernel in background; sample clock + power during sustained run
timeout 60 ./QuickRunCUDA -f tests/bench_v7_k1_throttle.cu -t 256 -p -0 5000 -A 1024 -B 1024 -C 1048576 > /tmp/k1.out 2>&1 &
KERN_PID=$!
sleep 2

echo "During sustained FFMA workload:"
for i in {1..15}; do
    nvidia-smi --query-gpu=power.draw,clocks.current.graphics --format=csv,noheader,nounits -i 0
    sleep 0.3
done

wait $KERN_PID 2>/dev/null

echo "Restoring clock to 1500..."
sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
