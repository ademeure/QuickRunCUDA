// F2FP latency microbench — dependent CVT chains the compiler cannot elide.
//
// Each thread runs N_CHAINS independent chains. Each chain is a CHAIN_LENGTH
// long sequence of F2FP ops, each consuming the previous output.
//
// CHAIN_FORMAT encodes the chain:
//   0: f32 → f16 (XOR 1) → f32 (XOR 1) → f32 → f16 → ...
//      Every CVT has a 1-bit XOR interposed so the compiler cannot fold the
//      round-trip. 2 F2FP + 2 tiny logic ops per CHAIN_LENGTH step.
//      (We subtract the LOP3 latency from the measurement.)
//   1: f32 → bf16 (+XOR) → f32 (+XOR) → ...
//   2: f32→f16x2(+XOR)→f32 pair(+XOR)→... packed variant
//   3: f32 pair → e4m3x2 (XOR) → f16x2 (XOR) → 2x f32 → ... (4 F2FP per step)
//   4: f32 pair → e2m1x2 (XOR) → f16x2 (XOR) → 2x f32 → ...
//   5: f16x2 → e4m3x2 (XOR) → f16x2 → e5m2x2 (XOR) → f16x2 → ...
//      Pure packed narrow, no f32 intermediate (2 F2FP per step)
//   6: f32 → tf32 (XOR) → f32 pair → f16x2 (XOR) → unpack
//
// For each format OPS_PER_STEP is how many F2FP PTX instructions one UNROLL
// iteration emits; driver must pass PERF_N = ITERS * N_CHAINS * OPS_PER_STEP.

#ifndef N_CHAINS
#define N_CHAINS 1
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef CHAIN_FORMAT
#define CHAIN_FORMAT 0
#endif

// All asm volatile to prevent CSE / DCE across iterations.

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    float r[N_CHAINS];
    unsigned short h[N_CHAINS];
    unsigned short b[N_CHAINS];
    unsigned int  p[N_CHAINS];

    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        r[k] = (float)(threadIdx.x + k) * 0.01f + 1.0f;
        p[k] = 0x3C003C01u ^ (threadIdx.x + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if CHAIN_FORMAT == 0
                // f32 -> f16 -> XOR -> f32 -> XOR -> f32  (2 F2FP + 2 XOR per step)
                asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(h[k]) : "f"(r[k]));
                asm volatile("xor.b16 %0, %0, 1;" : "+h"(h[k]));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(r[k]) : "h"(h[k]));
                asm volatile("xor.b32 %0, %0, 0x00010000;" : "+r"(*(unsigned int*)&r[k]));
#elif CHAIN_FORMAT == 1
                asm volatile("cvt.rn.satfinite.bf16.f32 %0, %1;" : "=h"(b[k]) : "f"(r[k]));
                asm volatile("xor.b16 %0, %0, 1;" : "+h"(b[k]));
                asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(r[k]) : "h"(b[k]));
                asm volatile("xor.b32 %0, %0, 0x00010000;" : "+r"(*(unsigned int*)&r[k]));
#elif CHAIN_FORMAT == 2
                // Packed: f32 pair -> f16x2 -> f32 pair.  2 F2FP per step (1 forward packed, 1 reverse via mov+2*cvt)
                // Simpler: alternate f32->f16->f32 but with thread-specific mask
                asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(h[k]) : "f"(r[k]));
                asm volatile("xor.b16 %0, %0, %1;" : "+h"(h[k]) : "h"((unsigned short)1));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(r[k]) : "h"(h[k]));
#elif CHAIN_FORMAT == 3
                // Packed narrow round-trip: f32 pair -> e4m3x2 (XOR) -> f16x2 (XOR) -> 2 f32 (add)
                float rr = r[k] + 0.5f;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(h[k]) : "f"(r[k]), "f"(rr));
                asm volatile("xor.b16 %0, %0, 1;" : "+h"(h[k]));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
                float lo, hi;
                asm volatile("{.reg .b16 l, hh; mov.b32 {l, hh}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, hh;}"
                             : "=f"(lo), "=f"(hi) : "r"(p[k]));
                r[k] = lo + hi;  // 2 F2FP from unpack + 1 FADD + 1 e4m3x2 + 1 f16x2.e4m3x2 = 4 F2FP per step
#elif CHAIN_FORMAT == 4
                // Same as 3 but e2m1x2 (FP4, lossier — .b8 output needs pack)
                float rr = r[k] + 0.5f;
                asm volatile("{ .reg .b8 _b; cvt.rn.satfinite.e2m1x2.f32 _b, %2, %1; mov.b16 %0,{_b,0}; }"
                             : "=h"(h[k]) : "f"(r[k]), "f"(rr));
                asm volatile("xor.b16 %0, %0, 1;" : "+h"(h[k]));
                asm volatile("{ .reg .b8 _b; mov.b16 {_b,_}, %1; cvt.rn.f16x2.e2m1x2 %0, _b; }"
                             : "=r"(p[k]) : "h"(h[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
                float lo, hi;
                asm volatile("{.reg .b16 l, hh; mov.b32 {l, hh}, %2; cvt.f32.f16 %0, l; cvt.f32.f16 %1, hh;}"
                             : "=f"(lo), "=f"(hi) : "r"(p[k]));
                r[k] = lo + hi;
#elif CHAIN_FORMAT == 5
                // Pure packed narrow: f16x2 -> e4m3x2 (XOR) -> f16x2 -> e5m2x2 (XOR) -> f16x2
                // 2 F2FP per step (forward + reverse through DIFFERENT narrow type)
                unsigned int pin = p[k];
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h[k]) : "r"(pin));
                asm volatile("xor.b16 %0, %0, 1;" : "+h"(h[k]));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
                // Second half: e5m2x2 round trip
                asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(b[k]) : "r"(p[k]));
                asm volatile("xor.b16 %0, %0, 1;" : "+h"(b[k]));
                asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(p[k]) : "h"(b[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
                // 4 F2FP per step
#elif CHAIN_FORMAT == 6
                // Pure forward: FP4 packed e2m1x2 <-> f16x2 chain
                unsigned int pin = p[k];
                asm volatile("{ .reg .b8 _b; cvt.rn.satfinite.e2m1x2.f16x2 _b, %1; mov.b16 %0,{_b,0}; }"
                             : "=h"(h[k]) : "r"(pin));
                asm volatile("xor.b16 %0, %0, 1;" : "+h"(h[k]));
                asm volatile("{ .reg .b8 _b; mov.b16 {_b,_}, %1; cvt.rn.f16x2.e2m1x2 %0, _b; }"
                             : "=r"(p[k]) : "h"(h[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
                // 2 F2FP per step
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        acc ^= __float_as_int(r[k]) ^ p[k] ^ (unsigned int)h[k] ^ (unsigned int)b[k];
    }
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
