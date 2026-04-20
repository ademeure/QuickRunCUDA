#!/usr/bin/env bash
# Popcount density sweep: L1, L2, DRAM-1G, DRAM-8G.
cd /root/github/QuickRunCUDA

DENSITIES=(0 1 2 4 8 12 16 20 24 28 30 31 32)

THREADS=512
BLOCKS=148

run_tier() {
  local TIER="$1"
  local KERNEL="$2"
  local WS_ARG="$3"
  local A="$4"
  local ITERS="$5"
  local RAMP="$6"
  local INT="$7"
  local EXTRA_H="$8"
  local OUT="/tmp/popcount_${TIER}.log"

  : > $OUT
  echo "# tier=${TIER} kernel=${KERNEL} ws_arg=${WS_ARG} A=${A} iters=${ITERS} extra_H='${EXTRA_H}'" | tee -a $OUT
  printf "%-3s %-7s %-7s %-7s %-7s %-7s %-7s %-7s\n" "d" "med_W" "act_W" "s1" "s2" "s3" "s4" "s5" | tee -a $OUT

  REUSE=""
  for d in "${DENSITIES[@]}"; do
    if [ -n "$EXTRA_H" ]; then
      ./QuickRunCUDA tests/${KERNEL} $REUSE \
        -t $THREADS -b $BLOCKS -i \
        -A $A -B 1 -C 1 \
        -0 $ITERS -1 $d -2 $WS_ARG \
        -H "$EXTRA_H" \
        > /tmp/pc4_${TIER}_d${d}.log 2>&1 &
    else
      ./QuickRunCUDA tests/${KERNEL} $REUSE \
        -t $THREADS -b $BLOCKS -i \
        -A $A -B 1 -C 1 \
        -0 $ITERS -1 $d -2 $WS_ARG \
        > /tmp/pc4_${TIER}_d${d}.log 2>&1 &
    fi
    PID=$!
    REUSE=""  # disabled - cubin-mismatch hazard

    sleep $RAMP
    s=()
    for k in 1 2 3 4 5; do
      pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
      s+=("$pw")
      sleep $INT
    done
    wait $PID 2>/dev/null

    med=$(printf "%s\n%s\n%s\n" "${s[1]}" "${s[2]}" "${s[3]}" | sort -g | sed -n '2p')
    act=$(awk "BEGIN{printf \"%.1f\", $med - 150}")
    printf "%-3s %-7.1f %-7s %-7s %-7s %-7s %-7s %-7s\n" \
      "$d" "$med" "$act" "${s[0]}" "${s[1]}" "${s[2]}" "${s[3]}" "${s[4]}" | tee -a $OUT
  done
  echo "Done ${TIER}." | tee -a $OUT
}

# L1: 64 KB per block, 100M iters → ~6s
run_tier "l1"      "bench_pwr_l1_popcount.cu"       65536      16777216  100000000 1.5 0.5 ""
# L2: 64 MB ws, 200M iters → ~6s sustained
run_tier "l2"      "bench_pwr_l2_popcount.cu"       67108864   16777216  200000000 1.5 0.5 ""
# DRAM-1G: ~5s sustained at ~7 TB/s
run_tier "dram1g"  "bench_pwr_dram_popcount.cu"     1073741824 268435456 100000000 1.5 0.5 ""
# DRAM-8G: needs compile-time WS_BYTES; ~6s at 7 TB/s
run_tier "dram8g"  "bench_pwr_dram_popcount64.cu"   0          2147483648 60000000 1.5 0.5 "#define WS_BYTES (8ULL*1024ULL*1024ULL*1024ULL)"

echo "ALL DONE"
