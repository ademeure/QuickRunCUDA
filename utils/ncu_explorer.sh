#!/bin/bash
# ncu_explorer.sh — explore available ncu metrics on B300
#
# Modes:
#   list <pattern>   List all metrics matching pattern (e.g. "l1tex|smem")
#   run <pattern> <command> Run command with metrics matching pattern
#   common <command> Run with the most useful metric set
#
# Examples:
#   ./ncu_explorer.sh list "l1tex.*hit"
#   ./ncu_explorer.sh run "dram__bytes" ./QuickRunCUDA -f tests/foo.cu
#   ./ncu_explorer.sh common ./QuickRunCUDA -f tests/foo.cu

set -uo pipefail

NCU=/usr/local/cuda/bin/ncu

if [ $# -lt 1 ]; then
    echo "Usage: $0 list <pattern>"
    echo "       $0 run <pattern> <command...>"
    echo "       $0 common <command...>"
    exit 1
fi

ACTION=$1
shift

case "$ACTION" in
    list)
        PATTERN=$1
        sudo -n $NCU --query-metrics 2>&1 | grep -iE "$PATTERN" | head -30
        ;;
    run)
        PATTERN=$1
        shift
        # Build comma-separated metric list (max 8)
        METRICS=$(sudo -n $NCU --query-metrics 2>&1 | grep -iE "$PATTERN" | awk '{print $1}' | head -8 | tr '\n' ',' | sed 's/,$//')
        echo "Metrics: $METRICS"
        sudo -n $NCU --metrics "$METRICS" --csv "$@" 2>&1 | tail -20
        ;;
    common)
        # Most-used metrics for B300 microbenches
        METRICS="dram__bytes_read.sum,dram__bytes_write.sum,l1tex__t_sector_hit_rate.pct,sm__cycles_elapsed.avg,smsp__inst_executed.sum"
        echo "Common metrics: $METRICS"
        sudo -n $NCU --metrics "$METRICS" --csv "$@" 2>&1 | grep -E "metrics|^[\"0]" | head -30
        ;;
    *)
        echo "Unknown action: $ACTION"
        exit 1
        ;;
esac
