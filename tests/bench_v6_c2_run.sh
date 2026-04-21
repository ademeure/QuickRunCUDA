#!/bin/bash
# V6 C2: rigorous DRAM-bound min-energy clock sweep
set -e

CLOCKS=(510 800 1005 1200 1500 1700 1992)
NSMS=148
THREADS=256
N_INNER=16384  # float4 reads per thread per outer iter
BYTES_PER_OUTER=$((NSMS * THREADS * N_INNER * 16))  # = 9.6 GB

ITERS_OUTER=400  # ~1 sec at 1500 MHz

echo "RIGOR PROTOCOL — V6 C2 DRAM-bound min-energy sweep"
echo "Per outer iter: $((BYTES_PER_OUTER / 1024 / 1024)) MB read"
echo ""

printf "%-7s %-9s %-8s %-9s %-12s %-13s %-12s\n" "CLK" "Pavg(W)" "Pact(W)" "time(ms)" "BW(GB/s)" "pJ/byte" "%peak"
echo "----------------------------------------------------------------------"

for CLK in "${CLOCKS[@]}"; do
    sudo nvidia-smi -i 0 -lgc $CLK,$CLK > /dev/null 2>&1
    sleep 1

    BASE_W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0)
    sleep 0.2

    timeout 60 ./QuickRunCUDA -f tests/bench_v6_c2_dram_energy.cu -t $THREADS -p \
        -0 $ITERS_OUTER -A 67108864 -B 1024 -C 1048576 -T 3 > /tmp/c2_kern.out 2>&1 &
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

    KTIME_MS=$(grep -oE "[0-9.]+ ms$" /tmp/c2_kern.out | head -1 | awk '{print $1}')

    if [ -n "$KTIME_MS" ] && [ "$KTIME_MS" != "0" ]; then
        TOTAL_BYTES=$(python3 -c "print($BYTES_PER_OUTER * $ITERS_OUTER)")
        BW_GBPS=$(python3 -c "print(f'{$TOTAL_BYTES / ($KTIME_MS / 1000) / 1e9:.1f}')")
        PJ_PER_BYTE=$(python3 -c "print(f'{$ACT_P * ($KTIME_MS/1000) / $TOTAL_BYTES * 1e12:.3f}')")
        PCT_PEAK=$(python3 -c "print(f'{$BW_GBPS / 7500 * 100:.1f}')")
        printf "%-7s %-9s %-8s %-9s %-12s %-13s %-12s\n" "$CLK" "$AVG_P" "$ACT_P" "$KTIME_MS" "$BW_GBPS" "$PJ_PER_BYTE" "$PCT_PEAK"
    else
        echo "ERROR at clock $CLK"
        cat /tmp/c2_kern.out | tail -5
    fi
done

sudo nvidia-smi -i 0 -lgc 1500,1500 > /dev/null 2>&1
echo "(restored to 1500 MHz)"
