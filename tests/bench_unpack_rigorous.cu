// Rigorous UNPACK-only test to verify F2FP.UNPACK peak rate.
// No compiler tricks. Chain-unique XOR feedback prevents CSE.
// SASS must contain exactly N_CHAINS * UNROLL F2FP.UNPACK instructions.

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
                // UNPACK: e4m3x2 (u8x2 packed in u16) -> f16x2 (two halves packed in u32)
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"(h[k]));
                // Feedback with chain-unique constant (blocks CSE)
                h[k] = (unsigned short)(tmp ^ (0xAB00u + k * 7));
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= (unsigned int)h[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
