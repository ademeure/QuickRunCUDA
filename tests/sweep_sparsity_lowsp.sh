#!/usr/bin/env bash
# Low-sparsity tail: 1, 2, 5, 15, 20% for L2 (most interesting tier)
cd /root/github/QuickRunCUDA

GRANS=(1 4 32 128)
GRAN_NAMES=(byte dword 32B 128B)
SPARSITIES=(1 2 5 15 20)
THREADS=512
BLOCKS=148
ITERS=200000000
WS_BYTES=67108864

OUT=/tmp/sparsity_l2_lowsp.log
: > $OUT
echo "# L2 low-sparsity, val=zero only" | tee -a $OUT
printf "%-5s %-5s %-7s %-7s %-7s %-7s %-7s %-7s %-7s\n" "g" "sp%" "med_W" "act_W" "s1" "s2" "s3" "s4" "s5" | tee -a $OUT

REUSE=""
for gi in 0 1 2 3; do
  g="${GRANS[$gi]}"; gname="${GRAN_NAMES[$gi]}"
  for sp in "${SPARSITIES[@]}"; do
    packed=$(((sp << 16) | (g << 8) | 0))   # val = zero
    ./QuickRunCUDA tests/bench_l2_sparsity.cu $REUSE \
      -t $THREADS -b $BLOCKS -i \
      -A 16777216 -B 1 -C 1 \
      -0 $ITERS -1 $packed -2 $WS_BYTES \
      > /tmp/spls_l2_g${g}_sp${sp}.log 2>&1 &
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
    printf "%-5s %-5s %-7.1f %-7s %-7s %-7s %-7s %-7s %-7s\n" \
      "$gname" "$sp" "$med" "$act" "${s[0]}" "${s[1]}" "${s[2]}" "${s[3]}" "${s[4]}" | tee -a $OUT
  done
done
echo "Done." | tee -a $OUT
