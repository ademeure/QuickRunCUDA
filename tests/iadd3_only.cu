#ifndef UNROLL
#define UNROLL 16
#endif
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int x = 0xBEEFCAFE ^ threadIdx.x;
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            asm volatile("add.u32 %0, %0, 1;" : "+r"(x));
        }
    }
    if ((int)x == seed) ((unsigned int*)C)[threadIdx.x] = x;
}
