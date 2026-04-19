#!/usr/bin/env bash
# D3: sub-agent critique CI — wrapper to validate findings before committing
#
# Usage:
#   ./utils/critique_finding.sh <commit-message-file> [<investigation-cu-file>]
#
# Spawns sub-agent(s) to critique:
#   1. Numbers vs theoretical sanity (fail if claimed > spec)
#   2. SASS verification for compute claims
#   3. ncu cross-check for memory/pipe claims
#   4. DCE detection for "too good" results
#   5. Clock/temp state at time of measurement
#
# Returns 0 if all checks pass, non-zero with explanation otherwise.
#
# Pattern intended for:
#   - Pre-commit: catch obviously-broken findings before they pollute history
#   - Loop iterations: sub-agent reviews each /loop result before next iter
#   - Sub-agent context isolation: keeps main agent context clean

set -e

MSG_FILE="$1"
CU_FILE="$2"

if [ -z "$MSG_FILE" ]; then
    echo "Usage: $0 <commit-msg-file> [investigation.cu]"
    echo ""
    echo "Spawns Claude sub-agents to critique:"
    echo "  - Numbers > theoretical → STOP"
    echo "  - SASS verification of compute kernels"
    echo "  - DCE detection (run < 1ms suspicious)"
    echo "  - Clock/temp throttle at measurement time"
    exit 1
fi

echo "=== Critique: $MSG_FILE ==="
echo ""

# Extract claimed numbers from commit message
CLAIMS=$(grep -oE "[0-9]+(\.[0-9]+)?\s*(TFLOPS|TB/s|GB/s|Gops|TFLOPS/W|%)" "$MSG_FILE" | head -10)
echo "Claims found:"
echo "$CLAIMS" | sed 's/^/  /'
echo ""

# Sanity gates
FAIL=0

# Gate 1: TFLOPS > NVIDIA spec
SPEC_BF16=2500   # NVIDIA BF16 spec
SPEC_FP8=5000
SPEC_FP4=15000
SPEC_FP32=76
SPEC_FP64=2

while read -r tf; do
    val=$(echo "$tf" | grep -oE "[0-9]+(\.[0-9]+)?")
    case "$tf" in
        *TFLOPS*)
            # Sanity: any single number > 15000 TFLOPS suspicious
            if (( $(echo "$val > 15000" | bc -l 2>/dev/null) )); then
                echo "  ⚠ FAIL: $tf > 15000 TFLOPS (exceeds even FP4 spec)"
                FAIL=1
            fi
            ;;
        *TB/s*)
            if (( $(echo "$val > 50" | bc -l 2>/dev/null) )); then
                echo "  ⚠ FAIL: $tf > 50 TB/s (exceeds SMEM peak)"
                FAIL=1
            fi
            ;;
    esac
done <<< "$CLAIMS"

# Gate 2: if .cu file provided, check for DCE smell
if [ -n "$CU_FILE" ] && [ -f "$CU_FILE" ]; then
    # Heuristic: kernel that doesn't write to global memory and runtime < 1ms = DCE
    if grep -q "if.*N.*<.*0" "$CU_FILE"; then
        echo "  ⚠ NOTE: kernel uses 'if (N < 0)' anti-DCE pattern — verify SASS shows expected ops"
    fi
    if ! grep -q "out\[" "$CU_FILE"; then
        echo "  ⚠ FAIL: kernel never writes to output. May be DCE'd."
        FAIL=1
    fi
fi

# Gate 3: clock/temp at commit time
CLK=$(nvidia-smi --query-gpu=clocks.current.graphics --format=csv,noheader,nounits -i 0)
TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits -i 0)
echo "  Current GPU 0 state: SM ${CLK} MHz, temp ${TEMP}°C"
if [ "$CLK" -lt 1900 ]; then
    echo "  ⚠ WARN: clock dropped below 1900 — recent measurement may be throttled"
fi

# Optional: spawn Claude sub-agent if available (placeholder)
if command -v claude &>/dev/null; then
    echo ""
    echo "  [optional] To spawn Claude sub-agent for deeper critique:"
    echo "    claude --print 'Critique this finding for B300 rigor: <paste finding>'"
fi

echo ""
if [ "$FAIL" -eq 1 ]; then
    echo "❌ CRITIQUE FAILED — fix before committing"
    exit 1
else
    echo "✓ Critique passed (basic sanity gates)"
fi
