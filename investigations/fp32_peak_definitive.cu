// fp32_peak_definitive.cu — definitive FFMA peak benchmark for B300 (sm_103a)
//
// Methodology:
//   - ILP chains = ILP (compile-time, 1..32)
//   - BLOCK_SIZE controls warps/block; blocks = SM_COUNT × MB for warps/SM control
//   - Inner loop is fully unrolled (INNER FFMAs per chain per iter)
//   - Outer loop NOT unrolled (OUTER iters) — keeps ptxas from exceeding register budget
//   - Each accumulator v[k] = v[k] * y + v[k], where y is threadIdx-derived (runtime opaque)
//     This means every FFMA result feeds the next FFMA in the same chain → true RAW dependency
//   - Anti-DCE: runtime-opaque sum of ALL accumulators, conditional store on impossible predicate
//     The sum is xor-reduced to prevent compiler from folding; the compare uses arg0 from host.
//
// Default: ILP=8, INNER=128, OUTER=100 → 1024 FFMA/thread/outer-iter
//          BLOCK_SIZE=1024, MB=6 (CTAs/SM) → ~6144 threads/SM (3× oversubscription)
//
// Usage via QuickRunCUDA:
//   ./QuickRunCUDA investigations/fp32_peak_definitive.cu \
//       -t 1024 -b <148*MB> -T 5 \
//       -N $((128 * 100 * ILP * 2)) -P 1e12 -U TFLOPS \
//       -0 <seed> -1 <ILP> -2 <INNER>
//
// FLOP count: threads × ILP × INNER × OUTER × 2  (2 = mul+add per FMA)

#ifndef ILP
#define ILP 8
#endif
#ifndef INNER
#define INNER 128
#endif
#ifndef OUTER
#define OUTER 100
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

// Macro: one FFMA step per chain element
// v[k] = v[k] * y + v[k]
// This is a true RAW chain: output of FMA feeds back as both src and dst of next FMA.

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int seed, int ilp_arg, int inner_arg) {
    unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Runtime-varying initial values (different per thread) so compiler can't fold constants
    float v[ILP];
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        v[k] = __int_as_float((tid + k + 1) | 0x3f800000u);  // in [1.0, 2.0)
    }
    // multiplier: thread-derived, not compile-time constant — prevents LICM
    float y = __int_as_float(((tid & 0xFF) + 0x3f800001u));  // ~1.0 + tiny epsilon

    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            // Each iteration: ILP independent FFMA chains
            // Using inline PTX to guarantee FFMA emission (not FADD/FMUL split)
            #pragma unroll
            for (int k = 0; k < ILP; k++) {
                asm volatile("fma.rn.f32 %0,%0,%1,%0;" : "+f"(v[k]) : "f"(y));
            }
        }
    }

    // Anti-DCE: xor-fold all accumulators (bitwise, so no FP cancellation worry)
    // then store if the impossible predicate matches — compiler can't prove it false
    // because seed comes from kernel arg (runtime).
    unsigned acc = 0;
    #pragma unroll
    for (int k = 0; k < ILP; k++) {
        acc ^= __float_as_int(v[k]);
    }
    // Cast seed to unsigned for comparison — avoids sign issues
    if (acc == (unsigned)seed) {
        C[tid] = __int_as_float(acc);
    }
}
