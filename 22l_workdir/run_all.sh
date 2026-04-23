#!/bin/bash
# Run all 22l grid-sync benchmarks at several grid sizes; collect to results.txt
set -e
cd /root/github/QuickRunCUDA/22l_workdir

OUT=/root/github/QuickRunCUDA/22l_workdir/results.txt
> $OUT

echo "# 22l grid-sync sweep $(date)" >> $OUT
echo "# clock-locked target=1800 (effective: see clock samples)" >> $OUT

CLOCK_BEFORE=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits)
echo "# clock_before=$CLOCK_BEFORE MHz" >> $OUT

for grid in 8 32 64 132 148; do
  for prog in cg_sync atomic_counter ninja_A ninja_B ninja_C ninja_D; do
    echo "" >> $OUT
    echo "## prog=$prog grid=$grid" >> $OUT
    if [ -f ./bench_22l_$prog ]; then
      timeout 30 ./bench_22l_$prog $grid 200 128 >> $OUT 2>&1 || echo "ERROR or TIMEOUT" >> $OUT
    fi
    sleep 0.5
  done
done

CLOCK_AFTER=$(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader,nounits)
echo "" >> $OUT
echo "# clock_after=$CLOCK_AFTER MHz" >> $OUT
echo "DONE - results in $OUT"
