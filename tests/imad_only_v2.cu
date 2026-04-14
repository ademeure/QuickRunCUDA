#ifndef UNROLL
#define UNROLL 16
#endif
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int x = 0xBEEFCAFE ^ threadIdx.x;
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            asm volatile("mad.lo.u32 %0, %0, 3, %1;" : "+r"(x) : "r"(i+j));
        }
    }
    if ((int)x == seed) ((unsigned int*)C)[threadIdx.x] = x;
}
