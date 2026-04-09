#!/bin/bash
# Giant CVT format conversion sweep - tests all PTX CVT variants on SM 10.3a
# Probes each instruction, benchmarks if it compiles, reports as table
set +e  # Don't exit on error - we expect some instructions to fail
cd "$(dirname "$0")/.."
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

[ -f QuickRunCUDA ] || make -j 2>&1 | tail -1

ITERS=4096
THREADS=256
TIMED=50
UNROLL=8

echo "============================================================"
echo "  CVT Format Conversion Sweep - $(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)"
echo "  SM arch: sm_103a, CUDA 13.0"
echo "============================================================"
echo ""

# Helper: try to compile and benchmark a CVT instruction
# Usage: try_cvt "label" "header_defines" ops_per_iter
try_cvt() {
    local label="$1"
    local header="$2"
    local ops="$3"
    local perf_n=$(python3 -c "print(${ITERS} * ${ops} / 1e9)")

    local result
    result=$(./QuickRunCUDA tests/bench_cvt_generic.cu -H "$header" \
        -p -t $THREADS -0 $ITERS -T $TIMED -N $perf_n -U "GOps/s" 2>&1)

    if echo "$result" | grep -q "error\|FAIL\|fatal"; then
        printf "%-55s  UNSUPPORTED\n" "$label"
    else
        local gops=$(echo "$result" | grep "==>" | sed 's/.*==> \([0-9.]*\).*/\1/')
        local ms=$(echo "$result" | grep "==>" | sed 's/\([0-9.]*\) ms.*/\1/')
        printf "%-55s  %10s GOps/s  (%s ms)\n" "$label" "$gops" "$ms"
    fi
}

#####################################################################
# SECTION 1: Narrow format conversions (the interesting ones)
#####################################################################
echo "================================================================="
echo "  NARROW FORMAT CONVERSIONS (FP4/FP6/FP8)"
echo "================================================================="
echo ""

# --- x2 to-narrow from f32 ---
echo "--- To-narrow from f32 pair (cvt.rn.satfinite, 4 CVTs/iter) ---"
for fmt in e2m1x2 e4m3x2 e5m2x2 e2m3x2 e3m2x2; do
    for relu in "" ".relu"; do
        if [ "$fmt" = "e2m1x2" ]; then
            body='asm volatile("{ .reg .b8 t0,t1,t2,t3; cvt.rn.satfinite'$relu'.'$fmt'.f32 t0,%6,%4; cvt.rn.satfinite'$relu'.'$fmt'.f32 t1,%7,%5; cvt.rn.satfinite'$relu'.'$fmt'.f32 t2,%6,%5; cvt.rn.satfinite'$relu'.'$fmt'.f32 t3,%7,%4; mov.b16 %0,{t0,0}; mov.b16 %1,{t1,0}; mov.b16 %2,{t2,0}; mov.b16 %3,{t3,0}; }" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3));'
        else
            body='asm volatile("{cvt.rn.satfinite'$relu'.'$fmt'.f32 %0,%6,%4; cvt.rn.satfinite'$relu'.'$fmt'.f32 %1,%7,%5; cvt.rn.satfinite'$relu'.'$fmt'.f32 %2,%6,%5; cvt.rn.satfinite'$relu'.'$fmt'.f32 %3,%7,%4;}" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3));'
        fi
        header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
        try_cvt "cvt.rn.satfinite${relu}.${fmt}.f32" "$header" 4
    done
done

echo ""
echo "--- To-narrow from f16x2 (cvt.rn.satfinite, 4 CVTs/iter) ---"
for fmt in e2m1x2 e4m3x2 e5m2x2 e2m3x2 e3m2x2; do
    for relu in "" ".relu"; do
        if [ "$fmt" = "e2m1x2" ]; then
            body='asm volatile("{ .reg .b8 t0,t1,t2,t3; cvt.rn.satfinite'$relu'.'$fmt'.f16x2 t0,%4; cvt.rn.satfinite'$relu'.'$fmt'.f16x2 t1,%5; cvt.rn.satfinite'$relu'.'$fmt'.f16x2 t2,%6; cvt.rn.satfinite'$relu'.'$fmt'.f16x2 t3,%7; mov.b16 %0,{t0,0}; mov.b16 %1,{t1,0}; mov.b16 %2,{t2,0}; mov.b16 %3,{t3,0}; }" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "r"(ir0),"r"(ir1),"r"(ir2),"r"(ir3));'
        else
            body='asm volatile("{cvt.rn.satfinite'$relu'.'$fmt'.f16x2 %0,%4; cvt.rn.satfinite'$relu'.'$fmt'.f16x2 %1,%5; cvt.rn.satfinite'$relu'.'$fmt'.f16x2 %2,%6; cvt.rn.satfinite'$relu'.'$fmt'.f16x2 %3,%7;}" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "r"(ir0),"r"(ir1),"r"(ir2),"r"(ir3));'
        fi
        header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
        try_cvt "cvt.rn.satfinite${relu}.${fmt}.f16x2" "$header" 4
    done
done

echo ""
echo "--- To-narrow from bf16x2 (cvt.rn.satfinite, 4 CVTs/iter) ---"
for fmt in e2m1x2 e4m3x2 e5m2x2; do
    body='asm volatile("{cvt.rn.satfinite.'$fmt'.bf16x2 %0,%4; cvt.rn.satfinite.'$fmt'.bf16x2 %1,%5; cvt.rn.satfinite.'$fmt'.bf16x2 %2,%6; cvt.rn.satfinite.'$fmt'.bf16x2 %3,%7;}" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "r"(ir0),"r"(ir1),"r"(ir2),"r"(ir3));'
    header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
    try_cvt "cvt.rn.satfinite.${fmt}.bf16x2" "$header" 4
done

echo ""
echo "--- From-narrow to f16x2 (cvt.rn, 4 CVTs/iter) ---"
for fmt in e2m1x2 e4m3x2 e5m2x2 e2m3x2 e3m2x2; do
    for relu in "" ".relu"; do
        if [ "$fmt" = "e2m1x2" ]; then
            body='asm volatile("{ .reg .b8 t0,t1,t2,t3; mov.b16 {t0,_},%4; mov.b16 {t1,_},%5; mov.b16 {t2,_},%6; mov.b16 {t3,_},%7; cvt.rn'$relu'.f16x2.'$fmt' %0,t0; cvt.rn'$relu'.f16x2.'$fmt' %1,t1; cvt.rn'$relu'.f16x2.'$fmt' %2,t2; cvt.rn'$relu'.f16x2.'$fmt' %3,t3; }" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "h"(ih0),"h"(ih1),"h"(ih2),"h"(ih3));'
        else
            body='asm volatile("{cvt.rn'$relu'.f16x2.'$fmt' %0,%4; cvt.rn'$relu'.f16x2.'$fmt' %1,%5; cvt.rn'$relu'.f16x2.'$fmt' %2,%6; cvt.rn'$relu'.f16x2.'$fmt' %3,%7;}" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "h"(ih0),"h"(ih1),"h"(ih2),"h"(ih3));'
        fi
        header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
        try_cvt "cvt.rn${relu}.f16x2.${fmt}" "$header" 4
    done
done

echo ""
echo "--- From-narrow to bf16x2 (cvt.rn, 4 CVTs/iter) ---"
for fmt in e2m1x2 e4m3x2 e5m2x2 e2m3x2 e3m2x2; do
    if [ "$fmt" = "e2m1x2" ]; then
        body='asm volatile("{ .reg .b8 t0,t1,t2,t3; mov.b16 {t0,_},%4; mov.b16 {t1,_},%5; mov.b16 {t2,_},%6; mov.b16 {t3,_},%7; cvt.rn.bf16x2.'$fmt' %0,t0; cvt.rn.bf16x2.'$fmt' %1,t1; cvt.rn.bf16x2.'$fmt' %2,t2; cvt.rn.bf16x2.'$fmt' %3,t3; }" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "h"(ih0),"h"(ih1),"h"(ih2),"h"(ih3));'
    else
        body='asm volatile("{cvt.rn.bf16x2.'$fmt' %0,%4; cvt.rn.bf16x2.'$fmt' %1,%5; cvt.rn.bf16x2.'$fmt' %2,%6; cvt.rn.bf16x2.'$fmt' %3,%7;}" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "h"(ih0),"h"(ih1),"h"(ih2),"h"(ih3));'
    fi
    header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
    try_cvt "cvt.rn.bf16x2.${fmt}" "$header" 4
done

echo ""
echo "--- x4 Stochastic Rounding from f32 (cvt.rs, 4 CVTs/iter) ---"
for fmt in e2m1x4 e4m3x4 e5m2x4 e2m3x4 e3m2x4; do
    for relu in "" ".relu"; do
        if [ "$fmt" = "e2m1x4" ]; then
            body='asm volatile("{ .reg .b16 t0,t1,t2,t3; cvt.rs'$relu'.satfinite.'$fmt'.f32 t0,{%4,%5,%6,%7},%16; cvt.rs'$relu'.satfinite.'$fmt'.f32 t1,{%8,%9,%10,%11},%16; cvt.rs'$relu'.satfinite.'$fmt'.f32 t2,{%12,%13,%14,%15},%16; cvt.rs'$relu'.satfinite.'$fmt'.f32 t3,{%4,%7,%5,%6},%16; mov.b16 %0,t0; mov.b16 %1,t1; mov.b16 %2,t2; mov.b16 %3,t3; }" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3), "f"(f4),"f"(f5),"f"(f6),"f"(f7), "f"(f0+f1),"f"(f2+f3),"f"(f4+f5),"f"(f6+f7), "r"(rbits));'
        else
            body='asm volatile("{cvt.rs'$relu'.satfinite.'$fmt'.f32 %0,{%4,%5,%6,%7},%16; cvt.rs'$relu'.satfinite.'$fmt'.f32 %1,{%8,%9,%10,%11},%16; cvt.rs'$relu'.satfinite.'$fmt'.f32 %2,{%12,%13,%14,%15},%16; cvt.rs'$relu'.satfinite.'$fmt'.f32 %3,{%4,%7,%5,%6},%16;}" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3), "f"(f4),"f"(f5),"f"(f6),"f"(f7), "f"(f0+f1),"f"(f2+f3),"f"(f4+f5),"f"(f6+f7), "r"(rbits));'
        fi
        header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
        try_cvt "cvt.rs${relu}.satfinite.${fmt}.f32" "$header" 4
    done
done

#####################################################################
# SECTION 2: Standard FP conversions (f32<->f16, f32<->bf16, tf32)
#####################################################################
echo ""
echo "================================================================="
echo "  STANDARD FP CONVERSIONS (f16/bf16/tf32)"
echo "================================================================="
echo ""

echo "--- f32 -> f16 ---"
for rnd in rn rz; do
    for mod in "" ".relu" ".satfinite" ".relu.satfinite"; do
        body='asm volatile("{cvt.'$rnd$mod'.f16.f32 %0,%4; cvt.'$rnd$mod'.f16.f32 %1,%5; cvt.'$rnd$mod'.f16.f32 %2,%6; cvt.'$rnd$mod'.f16.f32 %3,%7;}" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3));'
        header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
        try_cvt "cvt.${rnd}${mod}.f16.f32" "$header" 4
    done
done

echo ""
echo "--- f32 -> bf16 ---"
for rnd in rn rz; do
    for mod in "" ".relu" ".satfinite" ".relu.satfinite"; do
        body='asm volatile("{cvt.'$rnd$mod'.bf16.f32 %0,%4; cvt.'$rnd$mod'.bf16.f32 %1,%5; cvt.'$rnd$mod'.bf16.f32 %2,%6; cvt.'$rnd$mod'.bf16.f32 %3,%7;}" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3));'
        header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
        try_cvt "cvt.${rnd}${mod}.bf16.f32" "$header" 4
    done
done

echo ""
echo "--- f32,f32 -> f16x2 ---"
for rnd in rn rz; do
    for mod in "" ".relu" ".satfinite" ".relu.satfinite"; do
        body='asm volatile("{cvt.'$rnd$mod'.f16x2.f32 %0,%5,%4; cvt.'$rnd$mod'.f16x2.f32 %1,%6,%4; cvt.'$rnd$mod'.f16x2.f32 %2,%7,%5; cvt.'$rnd$mod'.f16x2.f32 %3,%5,%7;}" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3));'
        header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
        try_cvt "cvt.${rnd}${mod}.f16x2.f32" "$header" 4
    done
done

echo ""
echo "--- f32,f32 -> bf16x2 ---"
for rnd in rn rz; do
    for mod in "" ".relu" ".satfinite" ".relu.satfinite"; do
        body='asm volatile("{cvt.'$rnd$mod'.bf16x2.f32 %0,%5,%4; cvt.'$rnd$mod'.bf16x2.f32 %1,%6,%4; cvt.'$rnd$mod'.bf16x2.f32 %2,%7,%5; cvt.'$rnd$mod'.bf16x2.f32 %3,%5,%7;}" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3));'
        header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
        try_cvt "cvt.${rnd}${mod}.bf16x2.f32" "$header" 4
    done
done

echo ""
echo "--- f32 -> tf32 ---"
for combo in "rna.satfinite" "rn.satfinite" "rz.satfinite" "rna.satfinite.relu" "rn.satfinite.relu" "rz.satfinite.relu"; do
    body='asm volatile("{cvt.'$combo'.tf32.f32 %0,%4; cvt.'$combo'.tf32.f32 %1,%5; cvt.'$combo'.tf32.f32 %2,%6; cvt.'$combo'.tf32.f32 %3,%7;}" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3));'
    header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
    try_cvt "cvt.${combo}.tf32.f32" "$header" 4
done

echo ""
echo "--- f16 -> f32 / bf16 -> f32 ---"
body='asm volatile("{cvt.f32.f16 %0,%4; cvt.f32.f16 %1,%5; cvt.f32.f16 %2,%6; cvt.f32.f16 %3,%7;}" : "=f"(f0),"=f"(f1),"=f"(f2),"=f"(f3) : "h"(ih0),"h"(ih1),"h"(ih2),"h"(ih3));'
header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
try_cvt "cvt.f32.f16" "$header" 4

body='asm volatile("{cvt.f32.bf16 %0,%4; cvt.f32.bf16 %1,%5; cvt.f32.bf16 %2,%6; cvt.f32.bf16 %3,%7;}" : "=f"(f0),"=f"(f1),"=f"(f2),"=f"(f3) : "h"(ih0),"h"(ih1),"h"(ih2),"h"(ih3));'
header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
try_cvt "cvt.f32.bf16" "$header" 4

#####################################################################
# SECTION 3: ue8m0x2 scale factor conversions
#####################################################################
echo ""
echo "================================================================="
echo "  UE8M0X2 SCALE FACTOR CONVERSIONS"
echo "================================================================="
echo ""

for combo in "rz.satfinite" "rz" "rp.satfinite" "rp"; do
    body='asm volatile("{cvt.'$combo'.ue8m0x2.bf16x2 %0,%4; cvt.'$combo'.ue8m0x2.bf16x2 %1,%5; cvt.'$combo'.ue8m0x2.bf16x2 %2,%6; cvt.'$combo'.ue8m0x2.bf16x2 %3,%7;}" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "r"(ir0),"r"(ir1),"r"(ir2),"r"(ir3));'
    header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
    try_cvt "cvt.${combo}.ue8m0x2.bf16x2" "$header" 4
done

for combo in "rz.satfinite" "rz" "rp.satfinite" "rp"; do
    body='asm volatile("{cvt.'$combo'.ue8m0x2.f32 %0,%5,%4; cvt.'$combo'.ue8m0x2.f32 %1,%6,%4; cvt.'$combo'.ue8m0x2.f32 %2,%7,%5; cvt.'$combo'.ue8m0x2.f32 %3,%5,%7;}" : "=h"(h0),"=h"(h1),"=h"(h2),"=h"(h3) : "f"(f0),"f"(f1),"f"(f2),"f"(f3));'
    header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
    try_cvt "cvt.${combo}.ue8m0x2.f32" "$header" 4
done

body='asm volatile("{cvt.rn.bf16x2.ue8m0x2 %0,%4; cvt.rn.bf16x2.ue8m0x2 %1,%5; cvt.rn.bf16x2.ue8m0x2 %2,%6; cvt.rn.bf16x2.ue8m0x2 %3,%7;}" : "=r"(r0),"=r"(r1),"=r"(r2),"=r"(r3) : "h"(ih0),"h"(ih1),"h"(ih2),"h"(ih3));'
header="#define CVT_BODY $body
#define CVT_OPS_PER_ITER 4
#define UNROLL ${UNROLL}"
try_cvt "cvt.rn.bf16x2.ue8m0x2" "$header" 4

echo ""
echo "================================================================="
echo "  SWEEP COMPLETE"
echo "================================================================="
echo ""
echo "SASS files saved to sass/ directory"
ls sass/*.sass 2>/dev/null | wc -l
echo " SASS files total"
