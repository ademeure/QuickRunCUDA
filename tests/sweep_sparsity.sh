#!/usr/bin/env bash
# L2/DRAM read power vs sparsity at varying granularity and replacement value.
cd /root/github/QuickRunCUDA

GRANS=(1 4 32 128)
GRAN_NAMES=(byte dword 32B 128B)
VALUES=(0 1 2)            # 0=zeros, 1=ones, 2=0x55
VAL_NAMES=(zero one alt55)
SPARSITIES=(0 5 10 25 50 75 90 95 100)

# Two regimes: L2-warm (32 MiB) and DRAM-cold (8 GiB)
WS_NAME=("$1")           # "warm" or "cold" via $1, default warm
if [ -z "$1" ]; then WS_NAME="warm"; fi

if [ "$WS_NAME" = "cold" ]; then
  WS_BYTES=$((8 * 1024 * 1024 * 1024))   # 8 GiB
  ITERS=10000000                          # ~7s at 7 TB/s DRAM
  RAMP_SLEEP=2.0
  SAMPLE_INT=0.6
else
  WS_BYTES=$((32 * 1024 * 1024))          # 32 MiB
  ITERS=30000000                          # ~2.3s at 16 TB/s L2
  RAMP_SLEEP=0.8
  SAMPLE_INT=0.3
fi

A_DWORDS=$((WS_BYTES / 4))
THREADS=512
BLOCKS=148

OUT=/tmp/sparsity_sweep_${WS_NAME}.log
: > $OUT
echo "# Sparsity sweep, ws=${WS_NAME} (${WS_BYTES} bytes), iters=${ITERS}, ramp=${RAMP_SLEEP}s, int=${SAMPLE_INT}s" | tee -a $OUT
printf "%-5s %-5s %-7s %-7s %-7s %-7s %-7s %-7s %-7s\n" "g" "v" "sp%" "med_W" "s1" "s2" "s3" "s4" "s5" | tee -a $OUT

REUSE=""
for gi in 0 1 2 3; do
  g="${GRANS[$gi]}"
  gname="${GRAN_NAMES[$gi]}"
  for vi in 0 1 2; do
    v="${VALUES[$vi]}"
    vname="${VAL_NAMES[$vi]}"
    for sp in "${SPARSITIES[@]}"; do
      gran_value=$((g << 8 | v))
      ./QuickRunCUDA tests/bench_l2_sparsity.cu $REUSE \
        -t $THREADS -b $BLOCKS -i \
        -A $A_DWORDS -B 1 -C 1 \
        -0 $sp -1 $gran_value -2 $WS_BYTES \
        > /tmp/sp_${WS_NAME}_g${g}_v${v}_sp${sp}.log 2>&1 &
      PID=$!
      REUSE="--reuse-cubin"

      sleep $RAMP_SLEEP
      s=()
      for k in 1 2 3 4 5; do
        pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
        s+=("$pw")
        sleep $SAMPLE_INT
      done
      wait $PID 2>/dev/null

      med=$(printf "%s\n%s\n%s\n" "${s[1]}" "${s[2]}" "${s[3]}" | sort -g | sed -n '2p')
      printf "%-5s %-5s %-7s %-7.1f %-7s %-7s %-7s %-7s %-7s\n" \
        "$gname" "$vname" "$sp" "$med" "${s[0]}" "${s[1]}" "${s[2]}" "${s[3]}" "${s[4]}" | tee -a $OUT
    done
  done
done

echo "Done." | tee -a $OUT
