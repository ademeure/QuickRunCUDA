#!/bin/bash
# auto_rigor.sh — automated rigor protocol for B300 microbenchmarks
#
# Runs the 10-rule rigor protocol:
#   1-2. Wall-clock measurement (clean_run)
#   3-4. ncu cross-check (common metrics)
#   5-6. SASS verification (sass_count)
#   7. Three-method reconciliation
#
# Usage: ./auto_rigor.sh <kernel.cu> [QuickRunCUDA args...]

if [ $# -lt 1 ]; then
    echo "Usage: $0 <kernel.cu> [args...]"
    echo "Example: $0 tests/bench_d6_no_reuse.cu -t 32 -b 1 -T 1 -0 1000 -2 7 -H \"#define NCHAINS 16\""
    exit 1
fi

KERNEL=$1
shift
ARGS="$@"

DIR=$(dirname "$0")

echo "==========================================================="
echo "AUTO-RIGOR PROTOCOL for $KERNEL"
echo "==========================================================="
echo ""
echo "[Phase 1: clean GPU + baseline power]"
$DIR/clean_run.sh ./QuickRunCUDA -f $KERNEL $ARGS 2>&1 | head -10
echo ""

echo "[Phase 2: ncu common metrics]"
$DIR/ncu_explorer.sh common ./QuickRunCUDA -f $KERNEL $ARGS 2>&1 | head -10
echo ""

echo "[Phase 3: SASS opcode count]"
LATEST=$(ls -t sass/$(basename $KERNEL .cu)*.sass 2>/dev/null | head -1)
if [ -n "$LATEST" ]; then
    $DIR/sass_count.sh $LATEST 2>&1 | head -15
fi

echo ""
echo "==========================================================="
echo "Rigor checklist:"
echo "  [ ] Theoretical value stated"
echo "  [ ] Measured value collected"
echo "  [ ] Ratio to theoretical computed"
echo "  [ ] Cross-verified with ncu metrics"
echo "  [ ] SASS-verified instruction counts"
echo "  [ ] HIGH/MED/LOW confidence assigned"
echo "==========================================================="
