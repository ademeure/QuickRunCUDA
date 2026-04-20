#!/usr/bin/env bash
# Bit-stride L2 power sweep with runtime pattern_mode arg.
# Each run compiles fresh (--reuse-cubin removed - was source of cubin-mismatch bugs).
# Each pattern: ~3s kernel, sample 6×0.5s after 1.0s ramp, drop first 2 + last 1, median of 3.
cd /root/github/QuickRunCUDA

PATTERNS=(0 1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 99)
NAMES=(zeros bit2 bit2x2 nibble byte short word2x dword2x dword4x dword8x dword16x dword32x dword64x dword128x dword256x random)

WS_BYTES=$((64 * 1024 * 1024))
A_DWORDS=$((WS_BYTES / 4))
ITERS=50000000          # ~3.7s sustained at 16 TB/s
THREADS=512
BLOCKS=148

OUT=/tmp/bitstride_sweep.log
: > $OUT

printf "%-6s %-6s %-6s %-6s %-6s %-6s %-6s %-10s\n" "p" "med_W" "s1" "s2" "s3" "s4" "s5" "name" | tee -a $OUT

# Each run compiles fresh (cubin-mismatch hazard)
REUSE=""
i=0
while [ $i -lt ${#PATTERNS[@]} ]; do
  p="${PATTERNS[$i]}"
  name="${NAMES[$i]}"
  i=$((i+1))

  ./QuickRunCUDA tests/bench_l2_bitstride.cu $REUSE \
    -t $THREADS -b $BLOCKS -i \
    -A $A_DWORDS -B 1 -C 1 \
    -0 $ITERS -1 $p -2 $WS_BYTES \
    > /tmp/bs_run_$p.log 2>&1 &
  RUN_PID=$!
  REUSE=""  # disabled - cubin-mismatch hazard

  sleep 1.8   # init + ramp to steady-state
  s=()
  for k in 1 2 3 4 5; do
    pw=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits | head -1)
    s+=("$pw")
    sleep 0.3
  done
  wait $RUN_PID 2>/dev/null

  # Drop first 1 + last 1 → keep middle 3 (s[1], s[2], s[3]); median
  med=$(printf "%s\n%s\n%s\n" "${s[1]}" "${s[2]}" "${s[3]}" | sort -g | sed -n '2p')

  printf "%-6s %-6.1f %-6s %-6s %-6s %-6s %-6s %-10s\n" \
    "$p" "$med" "${s[0]}" "${s[1]}" "${s[2]}" "${s[3]}" "${s[4]}" "$name" | tee -a $OUT
done

echo "Done." | tee -a $OUT
