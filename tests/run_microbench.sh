#!/bin/bash
# Microbenchmark suite: DRAM BW, FP32 FMA, CVT narrow format conversions
# Usage: bash tests/run_microbench.sh [--correctness-only] [--perf-only] [--sass]
set -e
cd "$(dirname "$0")/.."

# Select GPU (override with GPU_ID env var)
export CUDA_VISIBLE_DEVICES=${GPU_ID:-0}

DO_CORRECTNESS=1
DO_PERF=1
DO_SASS=0
for arg in "$@"; do
    case $arg in
        --correctness-only) DO_PERF=0 ;;
        --perf-only)        DO_CORRECTNESS=0 ;;
        --sass)             DO_SASS=1 ;;
    esac
done

# Build
if [ ! -f QuickRunCUDA ] || [ QuickRunCUDA.cpp -nt QuickRunCUDA ]; then
    echo "Building QuickRunCUDA..."
    make -j 2>&1 | tail -1
fi

NVDISASM=$(which nvdisasm 2>/dev/null || echo "/usr/local/cuda/bin/nvdisasm")
TIMED_RUNS=100

# GPU specs
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader -i 0)
MAX_SM_CLK=$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader,nounits -i 0 | tr -d ' ')
MAX_MEM_CLK=$(nvidia-smi --query-gpu=clocks.max.mem --format=csv,noheader,nounits -i 0 | tr -d ' ')

echo "============================================================"
echo "GPU: $GPU_NAME"
echo "Max SM: ${MAX_SM_CLK} MHz, Max Mem: ${MAX_MEM_CLK} MHz"
echo "============================================================"
echo ""

# Helper: dump SASS for the last compiled kernel
sass_dump() {
    if [ $DO_SASS -eq 1 ]; then
        echo "    --- SASS (first ~40 instruction lines) ---"
        $NVDISASM output.cubin --print-code 2>/dev/null | grep -E "^\s+/\*|F2FP|I2F|F2F|FFMA|LDG|STG|IADD|BRA|MOV" | head -40
        echo ""
    fi
}

# ============================================================
# CORRECTNESS
# ============================================================
if [ $DO_CORRECTNESS -eq 1 ]; then
    echo "========================================"
    echo "  Correctness Tests"
    echo "========================================"
    ./QuickRunCUDA tests/test_cvt_correctness.cu -t 1 -b 1
    echo ""
fi

# ============================================================
# PERF BENCHMARKS
# ============================================================
if [ $DO_PERF -eq 1 ]; then

# ---- 1) DRAM Bandwidth ----
echo "========================================"
echo "  1) DRAM Bandwidth"
echo "========================================"
ARRAY_DWORDS=$((64 * 1024 * 1024))
THREADS=256
BLOCKS=$((ARRAY_DWORDS / 4 / THREADS))
PERF_MULT=$(python3 -c "print(${ARRAY_DWORDS} * 4 * 2 / 1e9)")
SOL_BW=$(python3 -c "print(2 * ${MAX_MEM_CLK}e6 * 8192 / 8 / 1e9)")
echo "  ${BLOCKS} blocks x ${THREADS} threads, 256MB read + 256MB write"
echo "  Theoretical peak: ${SOL_BW} GB/s"
./QuickRunCUDA tests/bench_dram_bw.cu \
    -A $ARRAY_DWORDS -C $ARRAY_DWORDS \
    -t $THREADS -b $BLOCKS \
    -T $TIMED_RUNS -P $PERF_MULT -U "GB/s" -L $SOL_BW \
    --l2flush 1
sass_dump
echo ""

# ---- 2) FP32 FMA Throughput ----
echo "========================================"
echo "  2) FP32 FMA Throughput"
echo "========================================"
FMA_ITERS=8192
THREADS=256
PERF_N=$(python3 -c "print(${FMA_ITERS} * 8 * 2 / 1e12)")  # 8 FMAs/iter, 2 FLOPS/FMA
SM_COUNT=160  # B300 assumption
SOL_TFLOPS=$(python3 -c "print(${SM_COUNT} * 128 * 2 * ${MAX_SM_CLK}e6 / 1e12)")
echo "  Persistent blocks, ${THREADS} threads/block, ${FMA_ITERS} iters x 8 FMAs, UNROLL=8"
echo "  Theoretical peak (${SM_COUNT} SMs): ${SOL_TFLOPS} TFLOPS"
./QuickRunCUDA tests/bench_fp32_fma.cu \
    -H "#define UNROLL 8" \
    -p -t $THREADS -0 $FMA_ITERS \
    -T $TIMED_RUNS -N $PERF_N -U "TFLOPS" -L $SOL_TFLOPS
sass_dump
echo ""

# ---- 3) CVT Narrow Format Throughput ----
echo "========================================"
echo "  3) CVT Narrow Format Throughput"
echo "========================================"
CVT_ITERS=8192
THREADS=256
CVT_UNROLL=8

# Helper function for CVT benchmarks
run_cvt() {
    local label="$1"
    local kernel="$2"
    local header="$3"
    local cvt_per_iter="$4"

    header="#define UNROLL ${CVT_UNROLL}
${header}"
    local perf_n=$(python3 -c "print(${CVT_ITERS} * ${cvt_per_iter} / 1e9)")
    echo "  $label"
    if ! ./QuickRunCUDA "$kernel" -H "$header" \
        -p -t $THREADS -0 $CVT_ITERS \
        -T $TIMED_RUNS -N $perf_n -U "GOps/s" 2>/dev/null; then
        echo "    COMPILE FAILED"
        return
    fi
    sass_dump
}

echo ""
echo "--- To-narrow from f16x2 (16 CVTs/iter) ---"
run_cvt "e2m1x2.f16x2 (FP4/NVFP4)" \
    tests/bench_cvt_to_narrow_f16x2.cu \
    "#define CVT_ASM cvt.rn.satfinite.e2m1x2.f16x2
#define CVT_B8" 16

run_cvt "relu.e2m1x2.f16x2 (FP4+ReLU)" \
    tests/bench_cvt_to_narrow_f16x2.cu \
    "#define CVT_ASM cvt.rn.satfinite.relu.e2m1x2.f16x2
#define CVT_B8" 16

run_cvt "e4m3x2.f16x2 (FP8 E4M3)" \
    tests/bench_cvt_to_narrow_f16x2.cu \
    "#define CVT_ASM cvt.rn.satfinite.e4m3x2.f16x2" 16

run_cvt "e5m2x2.f16x2 (FP8 E5M2)" \
    tests/bench_cvt_to_narrow_f16x2.cu \
    "#define CVT_ASM cvt.rn.satfinite.e5m2x2.f16x2" 16

echo ""
echo "--- To-narrow from f32 pair (8 CVTs/iter) ---"
run_cvt "e2m1x2.f32 (FP4)" \
    tests/bench_cvt_to_narrow_f32.cu \
    "#define CVT_ASM cvt.rn.satfinite.e2m1x2.f32
#define CVT_B8" 8

run_cvt "e4m3x2.f32 (FP8 E4M3)" \
    tests/bench_cvt_to_narrow_f32.cu \
    "#define CVT_ASM cvt.rn.satfinite.e4m3x2.f32" 8

run_cvt "e5m2x2.f32 (FP8 E5M2)" \
    tests/bench_cvt_to_narrow_f32.cu \
    "#define CVT_ASM cvt.rn.satfinite.e5m2x2.f32" 8

run_cvt "e2m3x2.f32 (FP6 E2M3)" \
    tests/bench_cvt_to_narrow_f32.cu \
    "#define CVT_ASM cvt.rn.satfinite.e2m3x2.f32" 8

run_cvt "e3m2x2.f32 (FP6 E3M2)" \
    tests/bench_cvt_to_narrow_f32.cu \
    "#define CVT_ASM cvt.rn.satfinite.e3m2x2.f32" 8

echo ""
echo "--- From-narrow to f16x2 (16 CVTs/iter) ---"
run_cvt "f16x2.e2m1x2 (FP4->FP16)" \
    tests/bench_cvt_from_narrow.cu \
    "#define CVT_ASM cvt.rn.f16x2.e2m1x2
#define CVT_B8" 16

run_cvt "f16x2.e4m3x2 (FP8->FP16)" \
    tests/bench_cvt_from_narrow.cu \
    "#define CVT_ASM cvt.rn.f16x2.e4m3x2" 16

run_cvt "f16x2.e5m2x2 (FP8->FP16)" \
    tests/bench_cvt_from_narrow.cu \
    "#define CVT_ASM cvt.rn.f16x2.e5m2x2" 16

run_cvt "f16x2.e2m3x2 (FP6->FP16)" \
    tests/bench_cvt_from_narrow.cu \
    "#define CVT_ASM cvt.rn.f16x2.e2m3x2" 16

run_cvt "f16x2.e3m2x2 (FP6->FP16)" \
    tests/bench_cvt_from_narrow.cu \
    "#define CVT_ASM cvt.rn.f16x2.e3m2x2" 16

echo ""
echo "--- x4 Stochastic Rounding from f32 (cvt.rs, 4 CVTs/iter) ---"
X4_ITERS=8192
X4_PERF_N=$(python3 -c "print(${X4_ITERS} * 4 / 1e9)")

run_x4() {
    local label="$1"
    local header="$2"
    echo "  $label"
    if ! ./QuickRunCUDA tests/bench_cvt_x4_f32.cu -H "$header" \
        -p -t $THREADS -0 $X4_ITERS \
        -T $TIMED_RUNS -N $X4_PERF_N -U "GOps/s" 2>/dev/null; then
        echo "    COMPILE FAILED"
        return
    fi
    sass_dump
}

run_x4 "e2m1x4.f32 (FP4 SR)" "#define CVT_ASM cvt.rs.satfinite.e2m1x4.f32"
run_x4 "e2m1x4.f32+relu (FP4 SR)" "#define CVT_ASM cvt.rs.relu.satfinite.e2m1x4.f32"
run_x4 "e4m3x4.f32 (FP8 E4M3 SR)" "#define CVT_ASM cvt.rs.satfinite.e4m3x4.f32
#define CVT_B32"
run_x4 "e5m2x4.f32 (FP8 E5M2 SR)" "#define CVT_ASM cvt.rs.satfinite.e5m2x4.f32
#define CVT_B32"
run_x4 "e2m3x4.f32 (FP6 E2M3 SR)" "#define CVT_ASM cvt.rs.satfinite.e2m3x4.f32
#define CVT_B32"
run_x4 "e3m2x4.f32 (FP6 E3M2 SR)" "#define CVT_ASM cvt.rs.satfinite.e3m2x4.f32
#define CVT_B32"

echo ""
fi  # DO_PERF

echo "Done!"
