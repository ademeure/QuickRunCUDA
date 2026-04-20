#!/usr/bin/env bash
# L2 read power vs popcount density (0..32 bits set per dword).
cd /root/github/QuickRunCUDA

DENSITIES=(0 1 2 3 4 6 8 10 12 14 16 18 20 22 24 26 28 30 31 32)

WS_BYTES=$((64 * 1024 * 1024))
A_DWORDS=$((WS_BYTES / 4))
ITERS=50000000
THREADS=512
BLOCKS=148

OUT=/tmp/popcount_sweep.log
: > $OUT
printf "%-6s %-6s %-6s %-6s %-6s %-6s %-6s\n" "d" "med_W" "s1" "s2" "s3" "s4" "s5" | tee -a $OUT

REUSE=""
for d in "${DENSITIES[@]}"; do
  ./QuickRunCUDA tests/bench_l2_popcount.cu $REUSE \
    -t $THREADS -b $BLOCKS -i \
    -A $A_DWORDS -B 1 -C 1 \
    -0 $ITERS -1 $d -2 $WS_BYTES \
    > /tmp/pc_run_$d.log 2>&1 &
  PID=$!
  REUSE="--reuse-cubin"

  sleep 1.8
  s=()
  for k in 1 2 3 4 5; do
    pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    s+=("$pw")
    sleep 0.3
  done
  wait $PID 2>/dev/null

  med=$(printf "%s\n%s\n%s\n" "${s[1]}" "${s[2]}" "${s[3]}" | sort -g | sed -n '2p')
  printf "%-6s %-6.1f %-6s %-6s %-6s %-6s %-6s\n" \
    "$d" "$med" "${s[0]}" "${s[1]}" "${s[2]}" "${s[3]}" "${s[4]}" | tee -a $OUT
done

echo "Done." | tee -a $OUT
