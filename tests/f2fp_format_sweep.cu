// Compare throughput of different F2FP variants
#ifndef FMT
#define FMT 0
#endif
#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned short h[N_CHAINS];
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        h[k] = 0x3C01 ^ (threadIdx.x + k);
        p[k] = 0x3C003C01u ^ (threadIdx.x + k);
    }
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if FMT == 0  /* e4m3 unpack */
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
#elif FMT == 1  /* e5m2 unpack */
                asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
#elif FMT == 2  /* e2m1 (FP4) unpack */
                asm volatile("{ .reg .b8 _b; mov.b16 {_b,_}, %1; cvt.rn.f16x2.e2m1x2 %0, _b; }"
                             : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
#elif FMT == 3  /* e2m3 (FP6) unpack */
                asm volatile("cvt.rn.f16x2.e2m3x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
#elif FMT == 4  /* e3m2 (FP6) unpack */
                asm volatile("cvt.rn.f16x2.e3m2x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
#elif FMT == 5  /* ue8m0 -> bf16x2 */
                asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(p[k]) : "h"(h[k]));
                h[k] = (unsigned short)p[k];
#elif FMT == 10  /* e4m3 pack */
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h[k]) : "r"(p[k]));
                p[k] = (unsigned int)h[k];
#elif FMT == 11  /* e5m2 pack */
                asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(h[k]) : "r"(p[k]));
                p[k] = (unsigned int)h[k];
#elif FMT == 12  /* e2m1 pack (FP4) */
                asm volatile("{ .reg .b8 _b; cvt.rn.satfinite.e2m1x2.f16x2 _b, %1; mov.b16 %0,{_b,0}; }"
                             : "=h"(h[k]) : "r"(p[k]));
                p[k] = (unsigned int)h[k];
#endif
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k] ^ (unsigned int)h[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
