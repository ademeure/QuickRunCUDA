#!/bin/bash
# Microbench template generator for QuickRunCUDA
# Usage: ./mkbench.sh <bench_name> [num_modes]
#   bench_name: e.g. "v5_x7_mymetric" — creates tests/bench_<name>.cu
#   num_modes:  number of MODE variants (default 2)
set -e

NAME=${1:?"Usage: $0 <name> [num_modes]"}
NMODES=${2:-2}
OUT="tests/bench_${NAME}.cu"

if [ -e "$OUT" ]; then
    echo "ERROR: $OUT already exists — choose a different name" >&2
    exit 1
fi

cat > "$OUT" <<EOF
// Microbench: ${NAME}
// TODO: describe what this measures and why
// MODE 0: baseline
// MODE 1: variant 1
// MODE 2: variant 2 (compare cost of each mode)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Input-dependent values (defeat compile-time constant folding)
    float a = (float)(threadIdx.x + 1) * 0.001f;
    float b = (float)(blockIdx.x + 1) * 0.001f;

    // Accumulator — MUST be used after the loop to prevent DCE
    float acc = 0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // TODO: baseline op — should depend on \`a\`, \`b\`, \`acc\`
        acc = acc * b + a;
#elif MODE == 1
        // TODO: variant 1
        acc = acc * b + a;
#elif MODE == 2
        // TODO: variant 2
        acc = acc * b + a;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Anti-DCE: write acc to C[] under an impossible condition
    if (acc == 1.234567e-30f) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d total_cy=%llu cy/iter=%.3f\n",
               MODE, ITERS, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
EOF

echo "Created: $OUT"
echo ""
echo "Next steps:"
echo "  1. Edit the MODE branches to implement the actual ops you want to measure"
echo "  2. Run:"
echo "     for M in 0 1 $( [ $NMODES -gt 2 ] && echo "2" ); do"
echo "       ./QuickRunCUDA -f $OUT -t 32 -b 1 -H \"#define MODE \$M\" -0 1000 \\"
echo "         -A 1024 -B 1024 -C 1024"
echo "     done"
echo "  3. SASS verify:  cuobjdump --dump-sass output.cubin | less"
echo "  4. ncu pipe dashboard:  ./utils/pipe_dashboard.sh ./QuickRunCUDA -f $OUT ..."
echo "  5. Rigor protocol: theoretical → measured → ratio → 3 methods → commit"
