#!/bin/bash
# sass_count.sh — count SASS opcodes in a kernel
#
# Counts instruction occurrences inside the main loop body (between L_x_1: and BRA back to it)
# Optionally filter by opcode regex.
#
# Usage: ./sass_count.sh <sass_file> [opcode_regex]
# Examples:
#   ./sass_count.sh sass/bench_d6_no_reuse_1234.sass
#   ./sass_count.sh sass/bench_d6_no_reuse_1234.sass "FFMA|HMMA"

set -uo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <sass_file> [opcode_regex]"
    echo ""
    echo "Available SASS files:"
    ls -1t sass/*.sass 2>/dev/null | head -10
    exit 1
fi

SASS=$1
PATTERN=${2:-".*"}

if [ ! -f "$SASS" ]; then
    echo "SASS file not found: $SASS"
    exit 1
fi

echo "=== Loop body in $SASS ==="
LOOP=$(sed -n '/L_x_1:/,/BRA.*L_x_1/p' "$SASS")
if [ -z "$LOOP" ]; then
    echo "(no main loop found, dumping all SASS opcodes)"
    LOOP=$(cat "$SASS")
fi

# Extract opcodes (first word after the addr comment)
OPCODES=$(echo "$LOOP" | grep -oE '/\*[0-9a-f]+\*/[[:space:]]+[A-Z][A-Z0-9_.]+' | awk '{print $NF}')

if [ -z "$OPCODES" ]; then
    echo "(no opcodes extracted)"
    exit 1
fi

# Filter by pattern, count, sort
COUNT=$(echo "$OPCODES" | grep -E "$PATTERN" | sort | uniq -c | sort -rn)
echo "$COUNT"
echo ""
echo "Total in loop: $(echo "$OPCODES" | wc -l)"
echo "Matched pattern: $(echo "$OPCODES" | grep -cE "$PATTERN")"
