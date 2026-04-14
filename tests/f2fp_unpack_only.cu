#ifndef UNROLL
#define UNROLL 16
#endif
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned short h = 0x3C01 ^ threadIdx.x;
    unsigned int p;
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p) : "h"(h));
            h = (unsigned short)p;
        }
    }
    if ((int)p == seed) ((unsigned int*)C)[threadIdx.x] = p;
}
