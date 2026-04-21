#!/bin/bash
# clean_run.sh — wrapper for clean B300 microbench runs
#
# Usage: ./clean_run.sh <command> [args...]
# Steps:
#   1. Kill any leftover QuickRunCUDA / standalone test processes
#   2. Wait for GPU to settle (sleep 5s)
#   3. Verify GPU clock is at expected lock state
#   4. Print baseline power
#   5. Run the command
#   6. Print final power
#
# Lessons baked in (per b300_clean memories):
# - Stuck procs silently inflate cy/op up to 8.5x
# - GPU clock can drift between runs without explicit lock
# - Need 5+ sec settle time for accurate baseline

if [ $# -lt 1 ]; then
    echo "Usage: $0 <command> [args...]"
    exit 1
fi

# Kill any leftover test processes
echo "[clean_run] Killing leftover test processes..."
# Kill by exact basename via pidof (avoids matching our own command line)
for proc in QuickRunCUDA h6_test h7_test r2_test r3_test r4_test l1_test l2_test; do
    PIDS=$(pidof -x $proc 2>/dev/null) || true
    [ -n "${PIDS:-}" ] && kill -9 $PIDS 2>/dev/null || true
done

# Settle
sleep 5

# Verify clock
CLOCK=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits | head -1)
echo "[clean_run] GPU clock: ${CLOCK} MHz"

# Baseline power (3 samples)
echo "[clean_run] Baseline power samples:"
for i in 1 2 3; do
    P=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    echo "  $P W"
    sleep 0.4
done

# Run the command
echo "[clean_run] Running: $@"
echo "==============="
"$@"
EXIT_CODE=$?
echo "==============="

# Final power
echo "[clean_run] Final power:"
sleep 2
P=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
echo "  $P W"

exit $EXIT_CODE
