// Rigorous PACK-only test. SASS: N_CHAINS * UNROLL of F2FP.SATFINITE...

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
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(tmp) : "r"(p[k]));
                p[k] = ((unsigned int)tmp) ^ (0xCD000000u + k * 11);
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
