#!/usr/bin/env bash
# Rigor harness shell wrapper:
#   rigor_run.sh <binary> [ncu-key-metric]
# Auto-runs:
#   1. Plain wall-clock (binary's own output)
#   2. ncu with key DRAM/pipe metrics
#   3. cuobjdump SASS instruction count summary
# All 3 outputs go to stdout for human comparison.

set -e

BIN="$1"
NCU_METRIC="${2:-dram__bytes.sum.per_second,smsp__inst_executed_pipe_fma.avg.pct_of_peak_sustained_active}"

if [ -z "$BIN" ] || [ ! -x "$BIN" ]; then
    echo "Usage: $0 <binary> [ncu-metric-csv]"
    echo "Default metric: $NCU_METRIC"
    exit 1
fi

echo "============================================================"
echo "  RIGOR: $BIN"
echo "============================================================"

# [0] Pre-flight: check GPU is clean (no other procs) + report clock state
echo
echo "[0] GPU state pre-flight:"
echo "------------------------------------------------------------"
PROCS=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader -i 0 | grep -v "^$" | wc -l)
if [ "$PROCS" -gt 0 ]; then
    echo "  ⚠ WARNING: $PROCS process(es) currently running on GPU 0:"
    nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv -i 0
    echo "  Continuing anyway (set RIGOR_STRICT=1 to abort)"
    if [ "${RIGOR_STRICT:-0}" = "1" ]; then exit 2; fi
else
    echo "  ✓ GPU 0 is idle (no processes)"
fi
nvidia-smi --query-gpu=clocks.current.graphics,clocks.current.memory,clocks.current.video,power.draw,temperature.gpu --format=csv,noheader -i 0 | sed 's/^/  clocks: /'

echo
echo "[1] Wall-clock + program output:"
echo "------------------------------------------------------------"
"$BIN"

echo
echo "[2] ncu metrics (DRAM + pipe utilization):"
echo "------------------------------------------------------------"
/usr/local/cuda/bin/ncu --metrics "$NCU_METRIC" --launch-skip 5 --launch-count 1 "$BIN" 2>/dev/null | grep -E "(byte/s|peak_sustained|second|active|elapsed|%)" | head -20

echo
echo "[3] SASS instruction census:"
echo "------------------------------------------------------------"
/usr/local/cuda/bin/cuobjdump --dump-sass "$BIN" 2>/dev/null | \
  grep -oE "^\s*/\*[0-9a-f]+\*/\s+[A-Z][A-Z0-9_.]+" | \
  awk '{print $2}' | sort | uniq -c | sort -rn | head -20

echo
echo "[4] Confidence assessment:"
echo "------------------------------------------------------------"
# Check post-run GPU state for thermal throttle
END_CLK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits -i 0)
END_TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 0)
echo "  Post-run: SM clock=${END_CLK} MHz, temp=${END_TEMP}°C"
if [ "$END_CLK" -lt 1900 ]; then
    echo "  ⚠ Clock dropped below 1900 MHz — possible thermal throttle. Confidence: LOW"
elif [ "$END_TEMP" -gt 80 ]; then
    echo "  ⚠ Temp ${END_TEMP}°C is high — investigate cooling. Confidence: MED"
else
    echo "  ✓ Clock stable (${END_CLK} MHz), temp OK (${END_TEMP}°C). Confidence: HIGH"
fi

echo
echo "============================================================"
echo "Reconcile: do the 3 methods agree on the kernel's behavior?"
echo "  - Wall-clock GB/s should be within 10%% of ncu byte/s"
echo "  - Pipe utilization should match measured TFLOPS / peak"
echo "  - Most-frequent SASS instructions should match the expected ops"
echo "============================================================"
