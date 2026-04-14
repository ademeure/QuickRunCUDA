// Round-trip PACK+UNPACK — no LOP3 feedback required because PACK's output
// feeds UNPACK, and UNPACK's output (u32 -> narrowed via low 16 bits) feeds
// next PACK naturally. Tests true pack pipe peak.

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
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++)
        v[k] = 0x3C003C01u ^ (threadIdx.x * 137 + k * 23);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short packed;
                unsigned int unpacked;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(packed) : "r"(v[k]));
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(unpacked) : "h"(packed));
                v[k] = unpacked;
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
