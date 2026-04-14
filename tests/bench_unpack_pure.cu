// Pure UNPACK: feedback reuses UNPACK's own output as next iter's input
// (no LOP3/XOR on ALU). Compiler can't CSE because each iter's input = last
// iter's output. Tests multiple FP8/FP4 unpack variants.

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef VARIANT
#define VARIANT 0   // 0=e4m3, 1=e5m2, 2=e2m3 (mxfp6), 3=e3m2, 4=e2m1 (mxfp4)
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned short h[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++)
        h[k] = (unsigned short)(0x3C00 + (threadIdx.x * 131 + k * 17));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int tmp;
#if VARIANT == 0
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"(h[k]));
#elif VARIANT == 1
                asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(tmp) : "h"(h[k]));
#elif VARIANT == 2
                asm volatile("cvt.rn.bf16x2.e2m3x2 %0, %1;" : "=r"(tmp) : "h"(h[k]));
#elif VARIANT == 3
                asm volatile("cvt.rn.bf16x2.e3m2x2 %0, %1;" : "=r"(tmp) : "h"(h[k]));
#elif VARIANT == 4
                asm volatile("cvt.rn.bf16x2.e2m1x2 %0, %1;" : "=r"(tmp) : "h"(h[k]));
#endif
                // No LOP3 — feed low 16 bits of tmp back in as next u16 input.
                // This is just a register-alias narrowing; should not emit ALU op.
                h[k] = (unsigned short)tmp;
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= (unsigned int)h[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
