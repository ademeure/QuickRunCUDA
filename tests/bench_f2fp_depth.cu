// Measure F2FP latency by chaining K dependent F2FP ops per step.
//
// Per step, one outer asm block emits K sequential F2FP instructions that
// genuinely depend on each other (alternating e4m3x2 ↔ f16x2 round-trips).
// Plus one integer XOR (LOP3) to defeat round-trip folding.
//
// Measured L_step(K) = K × L_f2fp + L_lop3  (single-warp-per-SM)
// so L_f2fp = L_step(K+1) − L_step(K)  for adjacent K.
//
// CHAIN_DEPTH: 1, 2, 3, 4 supported.
// N_CHAINS:    independent chains per thread (sweep to see saturation point).

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
#ifndef CHAIN_DEPTH
#define CHAIN_DEPTH 1
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        p[k] = 0x3C003C01u ^ (threadIdx.x + k);
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if CHAIN_DEPTH == 1
                // 1 F2FP: f16x2 → e4m3x2 (discard widen back via integer shuffle)
                // Then XOR closes the step
                unsigned short h;
                asm volatile(
                    "{ cvt.rn.satfinite.e4m3x2.f16x2 %0, %1; }"
                    : "=h"(h) : "r"(p[k]));
                // Widen via int-shuffle (not F2FP): duplicate the 16 bits into both halves of u32
                asm volatile("{ .reg .b16 hi; mov.b16 hi, %1; mov.b32 %0, {hi, hi}; }"
                    : "=r"(p[k]) : "h"(h));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
#elif CHAIN_DEPTH == 2
                // 2 F2FPs in one asm block: f16x2 → e4m3x2 → f16x2
                asm volatile(
                    "{ .reg .b16 _h;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %1;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                    : "=r"(p[k]) : "r"(p[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
#elif CHAIN_DEPTH == 3
                // 3 F2FPs: f16x2 → e4m3x2 → f16x2 → e5m2x2 (end as half16, widen via shuffle)
                unsigned short h;
                asm volatile(
                    "{ .reg .b16 _h1; .reg .b32 _p1;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h1, %1;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 _p1, _h1;\n\t"
                    "  cvt.rn.satfinite.e5m2x2.f16x2 %0, _p1; }"
                    : "=h"(h) : "r"(p[k]));
                asm volatile("{ .reg .b16 hi; mov.b16 hi, %1; mov.b32 %0, {hi, hi}; }"
                    : "=r"(p[k]) : "h"(h));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
#elif CHAIN_DEPTH == 4
                // 4 F2FPs: f16x2→e4m3x2→f16x2→e5m2x2→f16x2
                asm volatile(
                    "{ .reg .b16 _h1, _h2; .reg .b32 _p1;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h1, %1;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 _p1, _h1;\n\t"
                    "  cvt.rn.satfinite.e5m2x2.f16x2 _h2, _p1;\n\t"
                    "  cvt.rn.f16x2.e5m2x2 %0, _h2; }"
                    : "=r"(p[k]) : "r"(p[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
#elif CHAIN_DEPTH == 6
                // 6 F2FPs: f16x2 → e4m3 → f16x2 → e5m2 → f16x2 → e4m3 → f16x2
                asm volatile(
                    "{ .reg .b16 _h1, _h2, _h3; .reg .b32 _p1, _p2;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h1, %1;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 _p1, _h1;\n\t"
                    "  cvt.rn.satfinite.e5m2x2.f16x2 _h2, _p1;\n\t"
                    "  cvt.rn.f16x2.e5m2x2 _p2, _h2;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h3, _p2;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 %0, _h3; }"
                    : "=r"(p[k]) : "r"(p[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
#elif CHAIN_DEPTH == 8
                asm volatile(
                    "{ .reg .b16 _h1, _h2, _h3, _h4; .reg .b32 _p1, _p2, _p3;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h1, %1;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 _p1, _h1;\n\t"
                    "  cvt.rn.satfinite.e5m2x2.f16x2 _h2, _p1;\n\t"
                    "  cvt.rn.f16x2.e5m2x2 _p2, _h2;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h3, _p2;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 _p3, _h3;\n\t"
                    "  cvt.rn.satfinite.e5m2x2.f16x2 _h4, _p3;\n\t"
                    "  cvt.rn.f16x2.e5m2x2 %0, _h4; }"
                    : "=r"(p[k]) : "r"(p[k]));
                asm volatile("xor.b32 %0, %0, 0x00010001;" : "+r"(p[k]));
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
