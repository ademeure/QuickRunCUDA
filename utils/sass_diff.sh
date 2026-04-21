#!/bin/bash
# SASS diff visualizer — compare opcode counts between two cubin/SASS files
# Usage: ./sass_diff.sh <file1> <file2>
#   files can be .cubin or .sass; .cubin gets cuobjdump'd first

if [ $# -lt 2 ]; then
    echo "Usage: $0 <file1.{cubin,sass}> <file2.{cubin,sass}>" >&2
    exit 1
fi

extract_opcodes() {
    local F=$1
    if [[ "$F" == *.cubin ]]; then
        cuobjdump --dump-sass "$F" 2>/dev/null
    else
        cat "$F"
    fi | awk '
        # Extract first opcode after offset like /*0560*/
        /\/\*[0-9a-f]+\*\// {
            # Remove /*..*/ prefix and address
            sub(/^[ \t]*\/\*[0-9a-f]+\*\/[ \t]*/, "")
            # Get first token (opcode)
            split($0, parts, /[ \t]+/)
            opc = parts[1]
            # Strip predicate prefix like @P0, @!P0
            if (opc ~ /^@/) {
                opc = parts[2]
            }
            # Strip subop suffix like .E, .ADD, .STRONG.GPU — keep base mnemonic
            split(opc, base_parts, ".")
            base = base_parts[1]
            if (base != "") count[base]++
        }
        END {
            for (op in count) print count[op], op
        }' | sort -rn
}

TMP1=$(mktemp); TMP2=$(mktemp)
extract_opcodes "$1" > "$TMP1"
extract_opcodes "$2" > "$TMP2"

echo ""
echo "=================================================="
echo " SASS Opcode Diff"
echo " A: $1"
echo " B: $2"
echo "=================================================="
printf "%-15s %8s %8s %8s\n" "OPCODE" "A" "B" "Δ(B-A)"
echo "--------------------------------------------------"

# Combine both files' opcodes
all_opcodes=$( (cat "$TMP1" "$TMP2") | awk '{print $2}' | sort -u)

for OP in $all_opcodes; do
    A=$(grep -E "^[0-9]+ $OP$" "$TMP1" | awk '{print $1}')
    B=$(grep -E "^[0-9]+ $OP$" "$TMP2" | awk '{print $1}')
    A=${A:-0}
    B=${B:-0}
    DELTA=$((B - A))
    if [ "$DELTA" -ne 0 ] || [ "$A" -ne 0 ] || [ "$B" -ne 0 ]; then
        if [ "$DELTA" -gt 0 ]; then
            DELTA_STR="+$DELTA"
        else
            DELTA_STR="$DELTA"
        fi
        printf "%-15s %8d %8d %8s\n" "$OP" "$A" "$B" "$DELTA_STR"
    fi
done | sort -k4 -n -r

rm -f "$TMP1" "$TMP2"
echo "=================================================="
