#!/bin/bash
# After main sweep, probe clock dynamics during a long sustained run
LOGDIR=/root/github/QuickRunCUDA/justifications/clock_state_logs/sustained_probe
cd /root/github/QuickRunCUDA

# Wait for main sweep to finish
while pgrep -fa "run_sweep_v2.sh" > /dev/null; do sleep 5; done

sudo nvidia-smi -rgc; sleep 3

# Sample at 250ms (max nvidia-smi rate) for 20s, then start kernel
(timeout 65 nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu --format=csv,noheader,nounits -lms 250) > $LOGDIR/probe_clock_dynamics.csv 2>&1 &
SAMPLER=$!
sleep 5

echo "$(date +%H:%M:%S.%N) START KERNEL"

# Run a sustained kernel ~10 seconds
# T_RUNS=2000 * 0.44ms = 0.88s ... need more iters
timeout 30 ./QuickRunCUDA tests/bench_fp32_fma.cu -p -t 1024 -0 12800 -T 5000 -H "#define UNROLL 128" 2>&1 | tail -3 > $LOGDIR/kernel_output.txt
echo "$(date +%H:%M:%S.%N) END KERNEL"

wait $SAMPLER
echo "DONE_PROBE"
