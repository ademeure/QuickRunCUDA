#!/bin/bash
# Per-pipe utilization dashboard for B300 (sm_103a)
# Usage: ./pipe_dashboard.sh ./QuickRunCUDA -f tests/foo.cu [args...]
# Outputs: % peak sustained for each pipe (normalized 0-100%)

if [ $# -lt 1 ]; then
    echo "Usage: $0 <binary> [args...]" >&2
    exit 1
fi

# Pipes (per F1 commit 948cf45) — request normalized .avg.pct_of_peak_sustained_elapsed
PIPES="fma fmaheavy fmalite alu xu lsu tensor adu cbu fp64 tex"

METRIC_LIST=""
for PIPE in $PIPES; do
    METRIC_LIST+="sm__pipe_${PIPE}_cycles_active.avg.pct_of_peak_sustained_elapsed,"
done
METRIC_LIST="${METRIC_LIST}smsp__inst_executed.sum,sm__cycles_elapsed.sum"

OUTPUT=$(sudo -n /usr/local/cuda/bin/ncu --metrics "$METRIC_LIST" "$@" 2>&1)

# Header
echo ""
echo "=================================================="
echo "  B300 Per-Pipe Utilization Dashboard"
echo "=================================================="
printf "%-12s %10s\n" "PIPE" "% PEAK"
echo "--------------------------------------------------"

for PIPE in $PIPES; do
    METRIC_NAME="sm__pipe_${PIPE}_cycles_active.avg.pct_of_peak_sustained_elapsed"
    PCT=$(echo "$OUTPUT" | awk -v m="$METRIC_NAME" '
        $0 ~ m {
            for (i=NF; i>=1; i--) if ($i ~ /^[0-9]+\.[0-9]+$/ || $i == "0") { print $i; exit }
        }')
    if [ -n "$PCT" ] && [ "$PCT" != "0" ]; then
        BAR_LEN=$(echo "$PCT * 0.6" | bc 2>/dev/null | cut -d. -f1)
        if [ -n "$BAR_LEN" ] && [ "$BAR_LEN" -gt 0 ]; then
            BAR=$(printf "%${BAR_LEN}s" | tr ' ' '#')
        else
            BAR=""
        fi
        printf "%-12s %9s%% %s\n" "$PIPE" "$PCT" "$BAR"
    elif [ -n "$PCT" ]; then
        printf "%-12s %9s%%\n" "$PIPE" "$PCT"
    fi
done

echo "=================================================="
ELAPSED=$(echo "$OUTPUT" | awk '/cycles_elapsed/{
    for (i=NF; i>=1; i--) {
        v=$i; gsub(",","",v);
        if (v ~ /^[0-9]+$/) { print v; exit }
    }}')
INST=$(echo "$OUTPUT" | awk '/smsp__inst_executed.sum/{
    for (i=NF; i>=1; i--) {
        v=$i; gsub(",","",v);
        if (v ~ /^[0-9]+$/) { print v; exit }
    }}')
echo "Total cycles elapsed: $ELAPSED"
echo "Total warp insts: $INST"
echo ""
echo "Notes:"
echo "  - fma = parent of fmaheavy + fmalite (B300's 2 FFMA sub-pipes)"
echo "  - tensor = HMMA + IMMA + tcgen05 (Blackwell tensor)"
echo "  - alu/xu = INT/SFU/MUFU"
echo "  - lsu = global/shared mem ops"
