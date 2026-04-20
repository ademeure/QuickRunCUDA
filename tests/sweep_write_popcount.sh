#!/usr/bin/env bash
# L2 write power vs popcount density
cd /root/github/QuickRunCUDA

DENSITIES=(0 1 2 4 8 12 16 20 24 28 30 31 32)

THREADS=512
BLOCKS=148
ITERS=15000000
WS_BYTES=67108864
A_DWORDS=$((WS_BYTES / 4))

OUT=/tmp/popcount_l2_write.log
: > $OUT
echo "# L2 write popcount sweep, iters=${ITERS}, ws=${WS_BYTES}" | tee -a $OUT
printf "%-3s %-7s %-7s %-7s %-7s %-7s %-7s %-7s\n" "d" "med_W" "act_W" "s1" "s2" "s3" "s4" "s5" | tee -a $OUT

REUSE=""
for d in "${DENSITIES[@]}"; do
  ./QuickRunCUDA tests/bench_pwr_l2_popcount_write.cu $REUSE \
    -t $THREADS -b $BLOCKS -i \
    -A $A_DWORDS -B 1 -C 1 \
    -0 $ITERS -1 $d -2 $WS_BYTES \
    > /tmp/pcw_l2_d${d}.log 2>&1 &
  PID=$!
  REUSE=""  # disabled - cubin-mismatch hazard

  sleep 1.5
  s=()
  for k in 1 2 3 4 5; do
    pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    s+=("$pw")
    sleep 0.4
  done
  kill -9 $PID 2>/dev/null
  wait $PID 2>/dev/null
  sleep 0.3

  med=$(printf "%s\n%s\n%s\n" "${s[1]}" "${s[2]}" "${s[3]}" | sort -g | sed -n '2p')
  act=$(awk "BEGIN{printf \"%.1f\", $med - 150}")
  printf "%-3s %-7.1f %-7s %-7s %-7s %-7s %-7s %-7s\n" \
    "$d" "$med" "$act" "${s[0]}" "${s[1]}" "${s[2]}" "${s[3]}" "${s[4]}" | tee -a $OUT
done
echo "Done." | tee -a $OUT
