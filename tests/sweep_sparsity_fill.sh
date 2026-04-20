#!/usr/bin/env bash
# Fill in sp=30/35/40/45 between sp=25 and sp=50 for the read sparsity sweep.
cd /root/github/QuickRunCUDA

GRANS=(1 4 32 128)
GRAN_NAMES=(byte dword 32B 128B)
VALUES=(0 1 2)
VAL_NAMES=(zero one alt55)
SPARSITIES=(30 35 40 45)

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
  local OUT="/tmp/sparsity_${TIER}_fill.log"
  : > $OUT
  echo "# fill sp=30,35,40,45 tier=${TIER} kernel=${KERNEL} iters=${ITERS}" | tee -a $OUT
  printf "%-5s %-6s %-5s %-7s %-7s %-7s %-7s %-7s %-7s %-7s\n" "g" "v" "sp%" "med_W" "act_W" "s1" "s2" "s3" "s4" "s5" | tee -a $OUT

  REUSE=""
  for gi in 0 1 2 3; do
    g="${GRANS[$gi]}"
    gname="${GRAN_NAMES[$gi]}"
    for vi in 0 1 2; do
      v="${VALUES[$vi]}"
      vname="${VAL_NAMES[$vi]}"
      for sp in "${SPARSITIES[@]}"; do
        packed=$(((sp << 16) | (g << 8) | v))
        if [ -n "$EXTRA_H" ]; then
          ./QuickRunCUDA tests/${KERNEL} $REUSE -t $THREADS -b $BLOCKS -i \
            -A $A -B 1 -C 1 -0 $ITERS -1 $packed -2 $WS_ARG -H "$EXTRA_H" \
            > /tmp/spf_${TIER}_g${g}_v${v}_sp${sp}.log 2>&1 &
        else
          ./QuickRunCUDA tests/${KERNEL} $REUSE -t $THREADS -b $BLOCKS -i \
            -A $A -B 1 -C 1 -0 $ITERS -1 $packed -2 $WS_ARG \
            > /tmp/spf_${TIER}_g${g}_v${v}_sp${sp}.log 2>&1 &
        fi
        PID=$!
        REUSE="--reuse-cubin"

        sleep $RAMP
        s=()
        for k in 1 2 3 4 5; do
          pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
          s+=("$pw")
          sleep $INT
        done
        kill -9 $PID 2>/dev/null
        wait $PID 2>/dev/null
        sleep 0.3

        med=$(printf "%s\n%s\n%s\n" "${s[1]}" "${s[2]}" "${s[3]}" | sort -g | sed -n '2p')
        act=$(awk "BEGIN{printf \"%.1f\", $med - 150}")
        printf "%-5s %-6s %-5s %-7.1f %-7s %-7s %-7s %-7s %-7s %-7s\n" \
          "$gname" "$vname" "$sp" "$med" "$act" "${s[0]}" "${s[1]}" "${s[2]}" "${s[3]}" "${s[4]}" | tee -a $OUT
      done
    done
  done
  echo "Done ${TIER}." | tee -a $OUT
}

run_tier "l1"     "bench_pwr_l1_sparsity.cu"      65536      16777216  100000000 1.5 0.4 ""
run_tier "l2"     "bench_l2_sparsity.cu"          67108864   16777216  200000000 1.5 0.4 ""
run_tier "dram8g" "bench_pwr_dram_sparsity64.cu"  0          2147483648 60000000 1.5 0.4 "#define WS_BYTES (8ULL*1024ULL*1024ULL*1024ULL)"

echo "ALL FILL DONE"
