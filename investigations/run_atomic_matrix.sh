#!/bin/bash
# run_atomic_matrix.sh — sweep all (scope x ordering) OP values for atomic_scope_ordering.cu
# Uses nvcc directly since NVRTC doesn't support #elif chains

set -e
cd "$(dirname "$0")"

ARCH=sm_103a
ITERS=4096
NRUNS=5

# Lock GPU clock
nvidia-smi -lgc 2032 -i 0 > /dev/null 2>&1 || true
sleep 2

echo "GPU clock: $(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader -i 0 2>/dev/null)"

# Compile and run a single OP, print cycles/iter
run_op() {
    local op=$1
    local label=$2
    local exe="/tmp/atom_op${op}"

    nvcc -arch=${ARCH} -O3 \
        -DITERS=${ITERS} -DOP=${op} \
        -o "${exe}" atomic_scope_ordering_standalone.cu \
        2>/dev/null

    # Run NRUNS times, take min
    local min_cy=999999999
    for i in $(seq 1 ${NRUNS}); do
        cy=$("${exe}" 2>/dev/null)
        if [ -n "$cy" ] && [ "$cy" -lt "$min_cy" ] 2>/dev/null; then
            min_cy=$cy
        fi
    done

    local cy_per_iter=$(echo "scale=1; $min_cy / $ITERS" | bc)
    local ns=$(echo "scale=2; $cy_per_iter / 2.032" | bc)
    echo "$op $cy_per_iter $ns $label"
}

# Valid OPs (0-16 shared, 21-37 global; seq_cst 17-20, 38-41 not supported by ptxas)
VALID_OPS=(
    "0:smem:unqualified:baseline"
    "1:smem:relaxed:cta"
    "2:smem:relaxed:cluster"
    "3:smem:relaxed:gpu"
    "4:smem:relaxed:sys"
    "5:smem:acquire:cta"
    "6:smem:acquire:cluster"
    "7:smem:acquire:gpu"
    "8:smem:acquire:sys"
    "9:smem:release:cta"
    "10:smem:release:cluster"
    "11:smem:release:gpu"
    "12:smem:release:sys"
    "13:smem:acq_rel:cta"
    "14:smem:acq_rel:cluster"
    "15:smem:acq_rel:gpu"
    "16:smem:acq_rel:sys"
    "21:global:unqualified:baseline"
    "22:global:relaxed:cta"
    "23:global:relaxed:cluster"
    "24:global:relaxed:gpu"
    "25:global:relaxed:sys"
    "26:global:acquire:cta"
    "27:global:acquire:cluster"
    "28:global:acquire:gpu"
    "29:global:acquire:sys"
    "30:global:release:cta"
    "31:global:release:cluster"
    "32:global:release:gpu"
    "33:global:release:sys"
    "34:global:acq_rel:cta"
    "35:global:acq_rel:cluster"
    "36:global:acq_rel:gpu"
    "37:global:acq_rel:sys"
)

echo "OP mem ordering scope cy/iter ns"
for entry in "${VALID_OPS[@]}"; do
    IFS=: read op mem ordering scope <<< "$entry"
    run_op "$op" "${mem}:${ordering}:${scope}"
done
