#!/usr/bin/env bash
# Write power vs popcount density across L2 + DRAM-8G.
cd /root/github/QuickRunCUDA

DENSITIES=(0 1 2 4 8 12 16 20 24 28 30 31 32)

THREADS=512
BLOCKS=148

run_tier() {
  local TIER="$1"
  local KERNEL="$2"
  local A="$3"
  local ITERS="$4"
  local WS_ARG="$5"
  local EXTRA_H="$6"
  local OUT="/tmp/popcount_${TIER}_write.log"
  : > $OUT
  echo "# tier=${TIER} write kernel=${KERNEL}" | tee -a $OUT
  printf "%-3s %-7s %-7s %-7s %-7s %-7s %-7s %-7s\n" "d" "med_W" "act_W" "s1" "s2" "s3" "s4" "s5" | tee -a $OUT

  REUSE=""
  for d in "${DENSITIES[@]}"; do
    if [ -n "$EXTRA_H" ]; then
      ./QuickRunCUDA tests/${KERNEL} $REUSE \
        -t $THREADS -b $BLOCKS -i \
        -A $A -B 1 -C 1 \
        -0 $ITERS -1 $d -2 $WS_ARG \
        -H "$EXTRA_H" \
        > /tmp/wpc_${TIER}_d${d}.log 2>&1 &
    else
      ./QuickRunCUDA tests/${KERNEL} $REUSE \
        -t $THREADS -b $BLOCKS -i \
        -A $A -B 1 -C 1 \
        -0 $ITERS -1 $d -2 $WS_ARG \
        > /tmp/wpc_${TIER}_d${d}.log 2>&1 &
    fi
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
  echo "Done ${TIER}." | tee -a $OUT
}

run_tier "l2"     "bench_pwr_l2_popcount_write.cu"      16777216  15000000 67108864 ""
run_tier "dram8g" "bench_pwr_dram_popcount_write64.cu"  2147483648 25000000 0 "#define WS_BYTES (8ULL*1024ULL*1024ULL*1024ULL)"

echo "ALL WRITES DONE"
