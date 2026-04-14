// Pure PACK: feedback is zero-extension of u16 -> u32 (should be free).
// Tests whether PACK's true pipe peak is same as UNPACK (64) or half (32).

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
#define VARIANT 0  // 0=e4m3, 1=e5m2
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++)
        p[k] = 0x3C003C01u ^ (threadIdx.x * 137 + k * 23);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short tmp;
#if VARIANT == 0
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(tmp) : "r"(p[k]));
#elif VARIANT == 1
                asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(tmp) : "r"(p[k]));
#endif
                // Zero-extend u16 -> u32 (should be free cast).
                p[k] = (unsigned int)tmp;
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
