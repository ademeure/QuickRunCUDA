#!/bin/bash
# CLEAN clock state sweep for B300 SXM6 AC, GPU 0
# After verifying no other GPU procs running.
# For each lock state: capture pre/during/post nvidia-smi, ncu, wall time, power, temp
# Run kernel 3x for reproducibility, with sampler running concurrently

set -u
LOGDIR=/root/github/QuickRunCUDA/justifications/clock_state_logs
cd /root/github/QuickRunCUDA

KERNEL="tests/bench_fp32_fma.cu"
ITERS=12800
THREADS=1024
T_RUNS=30
HEADER="#define UNROLL 128"

TOTAL_FLOPS=$(python3 -c "print(1024*148*12800*8*2)")
echo "TOTAL_FLOPS per launch = $TOTAL_FLOPS"

# Verify no contention
if pgrep -f "QuickRunCUDA\|bench_2" > /dev/null; then
  echo "ERROR: competing GPU procs found, abort"
  pgrep -fa "QuickRunCUDA\|bench_2"
  exit 1
fi

run_one() {
    local LABEL=$1
    local LOCK_CMD=$2
    local OUT="$LOGDIR/${LABEL}.txt"
    echo "=== $LABEL ===" | tee "$OUT"
    echo "Lock cmd: $LOCK_CMD" | tee -a "$OUT"

    sudo nvidia-smi -rgc >> "$OUT" 2>&1
    sleep 3
    if [ -n "$LOCK_CMD" ]; then
        eval "sudo $LOCK_CMD" >> "$OUT" 2>&1
    fi
    sleep 3

    echo "--- pre-run nvidia-smi" | tee -a "$OUT"
    nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu,pstate --format=csv | tee -a "$OUT"

    # Background sampler at 1Hz for ~25s (covers all 3 runs)
    (for i in $(seq 1 25); do
        nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu --format=csv,noheader,nounits
        sleep 1
    done) > "$LOGDIR/${LABEL}_sampled.txt" 2>&1 &
    SAMPLER=$!

    echo "--- 3x runs (event-timed)" | tee -a "$OUT"
    for run in 1 2 3; do
        OUT_RUN=$(timeout 60 ./QuickRunCUDA "$KERNEL" -p -t $THREADS -0 $ITERS -T $T_RUNS -H "$HEADER" 2>&1)
        # Extract last "X.XXXXX ms" line
        TIME_MS=$(echo "$OUT_RUN" | grep -oE "^[0-9]+\.[0-9]+ ms" | head -1)
        echo "RUN $run: $TIME_MS" | tee -a "$OUT"
        sleep 1
    done

    wait $SAMPLER 2>/dev/null

    echo "--- post-run nvidia-smi" | tee -a "$OUT"
    nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu,pstate --format=csv | tee -a "$OUT"

    echo "--- sampled during runs (clocks.gr,clocks.mem,power,temp)" | tee -a "$OUT"
    cat "$LOGDIR/${LABEL}_sampled.txt" | tee -a "$OUT"

    echo "--- ncu pass for ground-truth gpc__cycles_elapsed" | tee -a "$OUT"
    sudo /usr/local/cuda/bin/ncu \
        --metrics gpc__cycles_elapsed.avg.per_second,sm__cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,gpc__cycles_elapsed.max \
        --target-processes all \
        timeout 60 ./QuickRunCUDA "$KERNEL" -p -t $THREADS -0 $ITERS -T 1 -H "$HEADER" 2>&1 | tee -a "$OUT" | tail -30

    echo "=== END $LABEL ===" | tee -a "$OUT"
    echo "" | tee -a "$OUT"
}

run_one "01_unlocked"      ""
run_one "02_lgc510"        "nvidia-smi -lgc 510"
run_one "03_lgc1005"       "nvidia-smi -lgc 1005"
run_one "04_lgc1500"       "nvidia-smi -lgc 1500"
run_one "05_lgc1800"       "nvidia-smi -lgc 1800"
run_one "06_lgc1920"       "nvidia-smi -lgc 1920"
run_one "07_lgc2032"       "nvidia-smi -lgc 2032"
run_one "08_lgc2032_2032"  "nvidia-smi -lgc 2032,2032"
run_one "09_lgc2031"       "nvidia-smi -lgc 2031"
run_one "10_lgc2033"       "nvidia-smi -lgc 2033"
run_one "11_lgc1920_1920"  "nvidia-smi -lgc 1920,1920"
run_one "12_lgc1800_1800"  "nvidia-smi -lgc 1800,1800"
run_one "13_lgc1942"       "nvidia-smi -lgc 1942"
run_one "14_lgc2050"       "nvidia-smi -lgc 2050"  # above advertised max

# Final: leave unlocked
sudo nvidia-smi -rgc

echo "DONE all sweeps."
