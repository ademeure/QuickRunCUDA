#!/bin/bash
# V6 C5: LLM kernel min-energy clock sweep
set -e

CLOCKS=(510 800 1005 1500 1992)
ITERS=20000

echo "RIGOR — V6 C5 LLM kernel min-energy sweep"
echo ""
printf "%-7s %-9s %-8s %-9s %-12s\n" "CLK" "Pavg(W)" "Pact(W)" "time(ms)" "energy(mJ)"
echo "--------------------------------------------------------"

for CLK in "${CLOCKS[@]}"; do
    sudo nvidia-smi -i 0 -lgc $CLK,$CLK > /dev/null 2>&1
    sleep 1
    BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
    sleep 0.2

    # Run kernel in background loop (relaunch repeatedly to keep GPU busy during sample)
    (for r in {1..20}; do
        ./QuickRunCUDA -f tests/bench_v6_c5_llm_energy.cu -t 256 -p \
            -0 $ITERS -A 67108864 -B 1024 -C 1048576 -T 1 > /tmp/c5_one.out 2>&1
    done) &
    LOOP_PID=$!

    sleep 1.0  # warmup

    # Sample power 6x
    PSUM=0
    for i in 1 2 3 4 5 6; do
        PSAMP=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
        PSUM=$(echo "$PSUM + $PSAMP" | bc)
        sleep 0.2
    done
    AVG_P=$(echo "scale=2; $PSUM / 6" | bc)
    ACT_P=$(echo "scale=2; $AVG_P - $BASE_W" | bc)

    wait $LOOP_PID 2>/dev/null

    KTIME_MS=$(grep -oE "[0-9.]+ ms$" /tmp/c5_one.out | head -1 | awk '{print $1}')
    if [ -n "$KTIME_MS" ] && [ "$KTIME_MS" != "0" ]; then
        ENERGY_MJ=$(python3 -c "print(f'{$ACT_P * $KTIME_MS:.1f}')")
        printf "%-7s %-9s %-8s %-9s %-12s\n" "$CLK" "$AVG_P" "$ACT_P" "$KTIME_MS" "$ENERGY_MJ"
    else
        echo "ERROR at clock $CLK"
    fi
done

sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
echo "(restored to 1500 MHz)"
