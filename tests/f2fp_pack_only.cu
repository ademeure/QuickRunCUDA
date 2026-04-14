#ifndef UNROLL
#define UNROLL 16
#endif
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int p = 0x3C003C01u ^ threadIdx.x;
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned short h;
            asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h) : "r"(p));
            p = (unsigned int)h;
        }
    }
    if ((int)p == seed) ((unsigned int*)C)[threadIdx.x] = p;
}
