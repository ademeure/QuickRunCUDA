#!/usr/bin/env bash
# Popcount density sweep across L1, L2, and DRAM tiers.
# Uses 3 separate kernel files with different cache hints + working sets.
cd /root/github/QuickRunCUDA

DENSITIES=(0 1 2 4 8 12 16 20 24 28 30 31 32)

THREADS=512
BLOCKS=148

# Per-tier configs: (kernel, ws_or_per_block_bytes, A_dwords, iters, ramp, sample_int, name)
run_tier() {
  local TIER="$1"
  local KERNEL="$2"
  local WS_ARG="$3"
  local A="$4"
  local ITERS="$5"
  local RAMP="$6"
  local INT="$7"
  local OUT="/tmp/popcount_${TIER}.log"

  : > $OUT
  echo "# tier=${TIER} kernel=${KERNEL} ws_arg=${WS_ARG} iters=${ITERS} ramp=${RAMP}s int=${INT}s" | tee -a $OUT
  printf "%-3s %-7s %-7s %-7s %-7s %-7s %-7s %-7s\n" "d" "med_W" "act_W" "s1" "s2" "s3" "s4" "s5" | tee -a $OUT

  REUSE=""
  for d in "${DENSITIES[@]}"; do
    ./QuickRunCUDA tests/${KERNEL} $REUSE \
      -t $THREADS -b $BLOCKS -i \
      -A $A -B 1 -C 1 \
      -0 $ITERS -1 $d -2 $WS_ARG \
      > /tmp/pc3_${TIER}_d${d}.log 2>&1 &
    PID=$!
    REUSE="--reuse-cubin"

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

# L1 tier: 64 KB per block, 100M iters → ~6s
run_tier "l1" "bench_pwr_l1_popcount.cu" 65536 16777216 100000000 1.5 0.5

# L2 tier: 64 MB ws, 80M iters → ~6s
run_tier "l2" "bench_pwr_l2_popcount.cu" 67108864 16777216 80000000 1.5 0.5

# DRAM tier: 1 GB ws (max int-safe), 30M iters → ~5s at 7 TB/s DRAM
run_tier "dram" "bench_pwr_dram_popcount.cu" 1073741824 268435456 30000000 1.5 0.5

echo "ALL DONE"
