#!/bin/bash
# Clock state sweep for B300 SXM6 AC, GPU 0
# For each lock state: capture pre/during/post nvidia-smi, ncu, wall time, power, temp
# Run kernel 3x for reproducibility

set -u
LOGDIR=/root/github/QuickRunCUDA/justifications/clock_state_logs
cd /root/github/QuickRunCUDA

# Kernel config (constant across all runs)
KERNEL="tests/bench_fp32_fma.cu"
ITERS=12800   # arg0
THREADS=1024
T_RUNS=30
HEADER="#define UNROLL 128"

# Compute total FLOPs for one kernel launch:
# 1024 thr/blk * 148 blk * 12800 iter * 8 fmas/unroll-iter * 2 ops/fma = ?
TOTAL_FLOPS=$(python3 -c "print(1024*148*12800*8*2)")
echo "TOTAL_FLOPS per launch = $TOTAL_FLOPS"

run_one() {
    local LABEL=$1
    local LOCK_CMD=$2
    local OUT="$LOGDIR/${LABEL}.txt"
    echo "=== $LABEL ===" | tee "$OUT"
    echo "Lock cmd: $LOCK_CMD" | tee -a "$OUT"

    # Apply lock
    sudo nvidia-smi -rgc >> "$OUT" 2>&1
    sleep 2
    if [ -n "$LOCK_CMD" ]; then
        eval "sudo $LOCK_CMD" >> "$OUT" 2>&1
    fi
    sleep 3

    # Pre clock
    echo "--- pre" | tee -a "$OUT"
    nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu,pstate --format=csv | tee -a "$OUT"

    # Background sampler at 1Hz for ~12s
    (for i in $(seq 1 12); do
        nvidia-smi --query-gpu=clocks.gr,power.draw,temperature.gpu --format=csv,noheader,nounits
        sleep 1
    done) > "$LOGDIR/${LABEL}_sampled.txt" 2>&1 &
    SAMPLER=$!

    # Run 3x for reproducibility
    echo "--- 3x runs" | tee -a "$OUT"
    for run in 1 2 3; do
        OUT_RUN=$(timeout 60 ./QuickRunCUDA "$KERNEL" -p -t $THREADS -0 $ITERS -T $T_RUNS -H "$HEADER" 2>&1)
        echo "RUN $run: $OUT_RUN" | tee -a "$OUT"
        # Extract last line "0.44105 ms (0.44254 ms including L2 flushes)"
        sleep 1
    done

    wait $SAMPLER 2>/dev/null

    # Post clock
    echo "--- post" | tee -a "$OUT"
    nvidia-smi --query-gpu=clocks.gr,clocks.mem,power.draw,temperature.gpu,pstate --format=csv | tee -a "$OUT"

    # Sampled clock summary
    echo "--- sampled during runs (clocks.gr,power,temp)" | tee -a "$OUT"
    cat "$LOGDIR/${LABEL}_sampled.txt" | tee -a "$OUT"

    # ncu pass for ground-truth gpc__cycles_elapsed.avg.per_second
    echo "--- ncu pass" | tee -a "$OUT"
    sudo /usr/local/cuda/bin/ncu \
        --metrics gpc__cycles_elapsed.avg.per_second,sm__cycles_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__cycles_active.avg.pct_of_peak_sustained_active \
        --target-processes all \
        timeout 60 ./QuickRunCUDA "$KERNEL" -p -t $THREADS -0 $ITERS -T 1 -H "$HEADER" 2>&1 | tee -a "$OUT" | tail -50

    echo "=== END $LABEL ===" | tee -a "$OUT"
}

# Sweep all lock states
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

# Cleanup: leave clock unlocked
sudo nvidia-smi -rgc

echo "DONE all sweeps."
