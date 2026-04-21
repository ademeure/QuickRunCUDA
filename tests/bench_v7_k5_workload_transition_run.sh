#!/bin/bash
# V7 K5: Workload transition clock behavior
# Run light → heavy → light workload, sample clock during transitions
sudo nvidia-smi -i 0 -rgc > /dev/null 2>&1
sleep 1
echo "Initial state:"
nvidia-smi --query-gpu=power.draw,clocks.current.graphics --format=csv,noheader,nounits -i 0

# Sample clock continuously during multi-phase workload
echo "" > /tmp/clk_trace.txt
(for i in {1..50}; do
    nvidia-smi --query-gpu=power.draw,clocks.current.graphics --format=csv,noheader,nounits -i 0 >> /tmp/clk_trace.txt
    sleep 0.1
done) &
SAMPLER_PID=$!

# Phase 1: light (idle) for 1 sec
sleep 1

# Phase 2: heavy FFMA for 2 sec
timeout 5 ./QuickRunCUDA -f tests/bench_v7_k1_throttle.cu -t 256 -p -0 2000 -A 1024 -B 1024 -C 1048576 > /dev/null 2>&1 &
KERN_PID=$!
sleep 2

# Phase 3: idle again for 1 sec
wait $KERN_PID 2>/dev/null
sleep 1

wait $SAMPLER_PID
echo "Sample trace (P W, Clk MHz):"
head -5 /tmp/clk_trace.txt
echo "..."
sed -n '8,18p' /tmp/clk_trace.txt
echo "..."
tail -5 /tmp/clk_trace.txt

sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
