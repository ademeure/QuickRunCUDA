#!/bin/bash
# V8 L3: per-kernel energy measurement tool
# Usage: ./utils/kernel_energy.sh <kernel.cu> [QuickRunCUDA args]
# Runs the kernel with -T 1 (we'll loop at shell level for steady-state),
# samples nvidia-smi power.draw during the run, reports:
#   time (ms), avg power (W), peak power (W), total energy (J)
#
# Works best when kernel runtime > 1.5 sec (so sampling captures steady-state).
# Hint: pass large -0 (iters) to extend runtime.

set -e
cd "$(dirname "$0")/.."

if [ $# -lt 1 ]; then
    echo "Usage: $0 <kernel.cu> [QuickRunCUDA flags]"
    echo "  e.g.  $0 tests/bench_v6_c1_ffma_energy.cu -t 256 -b 148 -0 2000"
    exit 1
fi

KERNEL="$1"
shift
EXTRA="$@"

if [ ! -f "$KERNEL" ]; then echo "Kernel not found: $KERNEL"; exit 1; fi
if [ ! -f ./QuickRunCUDA ]; then make -s; fi

pkill -9 QuickRunCUDA 2>/dev/null || true
sleep 3

# Sample idle power first (5 samples × 0.3 sec)
IDLE_SUM=0; IDLE_CNT=0
for i in 1 2 3 4 5; do
    pwr=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
    IDLE_SUM=$(echo "$IDLE_SUM + $pwr" | bc -l)
    IDLE_CNT=$((IDLE_CNT + 1))
    sleep 0.25
done
IDLE=$(echo "scale=1; $IDLE_SUM / $IDLE_CNT" | bc -l)
CLOCK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')

echo "=== Kernel energy measurement ==="
echo "Kernel: $KERNEL"
echo "Args:   $EXTRA"
echo "Clock:  $CLOCK MHz, idle power $IDLE W"
echo

# Launch kernel in background; sample power during
./QuickRunCUDA -f "$KERNEL" $EXTRA -T 1 > /tmp/ke_out.txt 2>&1 &
PID=$!

# Sample every 0.15 sec while kernel runs
PWR_LIST=""
MAX_SAMPLES=50
for i in $(seq 1 $MAX_SAMPLES); do
    if ! kill -0 $PID 2>/dev/null; then break; fi
    pwr=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits -i 0 | head -1 | tr -d ' ')
    PWR_LIST="$PWR_LIST $pwr"
    sleep 0.15
done
wait $PID

TIME_MS=$(grep -oP '\d+\.\d+' /tmp/ke_out.txt | head -1)
if [ -z "$TIME_MS" ]; then echo "ERROR: no time found"; exit 1; fi

# Skip first sample (ramp-up noise), use stable samples
STABLE_LIST=$(echo $PWR_LIST | cut -d' ' -f3-)
PWR_STATS=$(echo $STABLE_LIST | tr ' ' '\n' | awk '
    BEGIN {sum=0; n=0; max=0}
    $1 > 0 {sum+=$1; n++; if($1>max) max=$1}
    END {if (n>0) printf "%.1f %.1f %d", sum/n, max, n; else printf "0 0 0"}')
PWR_AVG=$(echo $PWR_STATS | awk '{print $1}')
PWR_MAX=$(echo $PWR_STATS | awk '{print $2}')
N_STABLE=$(echo $PWR_STATS | awk '{print $3}')

TIME_S=$(echo "scale=4; $TIME_MS / 1000" | bc)
ENERGY=$(echo "scale=2; $PWR_AVG * $TIME_S" | bc)
DELTA_POWER=$(echo "scale=1; $PWR_AVG - $IDLE" | bc)
COMPUTE_ENERGY=$(echo "scale=2; $DELTA_POWER * $TIME_S" | bc)

echo "--- Results ---"
printf "  Runtime:        %s ms\n" "$TIME_MS"
printf "  Avg power:      %s W (%d stable samples)\n" "$PWR_AVG" "$N_STABLE"
printf "  Peak power:     %s W\n" "$PWR_MAX"
printf "  Delta over idle:%s W\n" "$DELTA_POWER"
printf "  Total energy:   %s J\n" "$ENERGY"
printf "  Compute energy: %s J (delta × time)\n" "$COMPUTE_ENERGY"

# If user provided ops metric via env, compute per-op energy
if [ -n "$OPS" ]; then
    E_PER_OP=$(echo "scale=20; $ENERGY / $OPS" | bc -l)
    echo "  E/op:          $E_PER_OP J (OPS=$OPS)"
fi

echo
echo "Hint: set \$OPS env var to total operations for E/op calc."
echo "  e.g. OPS=\$((148*256*iters*8*2)) ./utils/kernel_energy.sh ..."

# Clean up
rm -f /tmp/ke_out.txt
