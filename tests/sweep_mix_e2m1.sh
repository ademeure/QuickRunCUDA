#!/bin/bash
# Instruction mix sweep: e2m1x2 CVTs mixed with various companion instructions
# Tests co-issue capability of F2FP with other functional units
set +e
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

[ -f QuickRunCUDA ] || make -j 2>&1 | tail -1

ITERS=8192
THREADS=256
TIMED=100
UNROLL=4

echo "============================================================"
echo "  Instruction Mix Sweep: e2m1x2 + Companion Instructions"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "  SM clocks: $(nvidia-smi --query-gpu=clocks.sm --format=csv,noheader -i 0)"
echo "============================================================"
echo ""

run_mix() {
    local label="$1"
    local n_cvt="$2"
    local n_comp="$3"
    local comp_header="$4"
    local total_ops=$((n_cvt + n_comp))
    [ $total_ops -eq 0 ] && return

    local perf_n=$(python3 -c "print(${ITERS} * ${total_ops} / 1e9)")
    local header="#define N_CVT ${n_cvt}
#define N_COMP ${n_comp}
#define UNROLL ${UNROLL}
${comp_header}"

    local result
    result=$(./QuickRunCUDA tests/bench_mix_e2m1.cu -H "$header" \
        -p -t $THREADS -0 $ITERS -T $TIMED -N $perf_n -U "GOps/s" 2>&1)

    if echo "$result" | grep -q "error\|fatal"; then
        printf "  %-12s  cvt=%-2d comp=%-2d  FAILED\n" "$label" "$n_cvt" "$n_comp"
        return
    fi

    local ms=$(echo "$result" | grep "==>" | sed 's/\([0-9.]*\) ms.*/\1/')
    local total_gops=$(echo "$result" | grep "==>" | sed 's/.*==> \([0-9.]*\).*/\1/')
    # Compute per-type throughput
    local cvt_gops="--"
    local comp_gops="--"
    [ $n_cvt -gt 0 ] && cvt_gops=$(python3 -c "print(f'{${ITERS}*${n_cvt}/($ms/1000)/1e9:.0f}')")
    [ $n_comp -gt 0 ] && comp_gops=$(python3 -c "print(f'{${ITERS}*${n_comp}/($ms/1000)/1e9:.0f}')")

    printf "  %-12s  cvt=%-2d comp=%-2d  %8s ms  cvt=%6s comp=%6s GOps/s\n" \
        "$label" "$n_cvt" "$n_comp" "$ms" "$cvt_gops" "$comp_gops"
}

# =====================================================================
# For each companion type, sweep CVT:COMP ratios
# =====================================================================

declare -A COMP_HEADERS
COMP_HEADERS[FFMA]='#define COMP_INIT float c0=1.0f+tid_f, c1=2.0f+tid_f, c2=3.0f+tid_f, c3=4.0f+tid_f; float ca=1.0000001f, cb=0.9999999f;
#define COMP_1 asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(c0) : "f"(ca),"f"(cb));
#define COMP_SINK acc ^= __float_as_int(c0)^__float_as_int(c1)^__float_as_int(c2)^__float_as_int(c3);'

COMP_HEADERS[FMUL]='#define COMP_INIT float c0=1.0f+tid_f, c1=2.0f+tid_f, c2=3.0f+tid_f, c3=4.0f+tid_f;
#define COMP_1 asm volatile("mul.rn.f32 %0,%0,%1;" : "+f"(c0) : "f"(c1));
#define COMP_SINK acc ^= __float_as_int(c0)^__float_as_int(c1)^__float_as_int(c2)^__float_as_int(c3);'

COMP_HEADERS[LOP3]='#define COMP_INIT unsigned int x0=tid^0xDEAD, x1=tid^0xBEEF, x2=tid^0xCAFE, x3=tid^0xBABE;
#define COMP_1 asm volatile("xor.b32 %0,%0,%1;" : "+r"(x0) : "r"(x1));
#define COMP_SINK acc ^= x0^x1^x2^x3;'

COMP_HEADERS[IADD3]='#define COMP_INIT unsigned int x0=tid, x1=tid+1, x2=tid+2, x3=tid+3;
#define COMP_1 asm volatile("add.u32 %0,%0,%1;" : "+r"(x0) : "r"(x1));
#define COMP_SINK acc ^= x0^x1^x2^x3;'

COMP_HEADERS[IMAD]='#define COMP_INIT unsigned int x0=tid, x1=tid+1, x2=3, x3=tid+3;
#define COMP_1 asm volatile("mad.lo.u32 %0,%1,%2,%0;" : "+r"(x0) : "r"(x1),"r"(x2));
#define COMP_SINK acc ^= x0^x1^x2^x3;'

COMP_HEADERS[MUFU]='#define COMP_INIT float c0=1.0f+tid_f, c1=2.0f+tid_f, c2=3.0f+tid_f, c3=4.0f+tid_f;
#define COMP_1 asm volatile("ex2.approx.f32 %0,%0;" : "+f"(c0));
#define COMP_SINK acc ^= __float_as_int(c0)^__float_as_int(c1)^__float_as_int(c2)^__float_as_int(c3);'

# Baseline: CVT only
echo "=== BASELINE: e2m1x2 CVTs only ==="
echo "  (companion=none)"
for nc in 1 2 4 8 12 16; do
    run_mix "CVT_ONLY" $nc 0 ""
done
echo ""

# For each companion type
for comp_type in FFMA FMUL LOP3 IADD3 IMAD MUFU; do
    echo "=== e2m1x2 + $comp_type ==="

    # Companion baseline
    run_mix "$comp_type" 0 4 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 0 8 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 0 16 "${COMP_HEADERS[$comp_type]}"
    echo "  ---"

    # Fixed 4 CVTs, varying companions
    run_mix "$comp_type" 4 0 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 4 1 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 4 2 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 4 4 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 4 8 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 4 12 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 4 16 "${COMP_HEADERS[$comp_type]}"
    echo "  ---"

    # Fixed 4 companions, varying CVTs
    run_mix "$comp_type" 1 4 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 2 4 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 4 4 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 8 4 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 12 4 "${COMP_HEADERS[$comp_type]}"
    run_mix "$comp_type" 16 4 "${COMP_HEADERS[$comp_type]}"
    echo ""
done

echo "=== SWEEP COMPLETE ==="
