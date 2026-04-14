#ifndef UNROLL
#define UNROLL 16
#endif
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float r = 1.0001f + (float)threadIdx.x * 0.0001f;
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(r));
        }
    }
    if (__float_as_int(r) == seed) C[threadIdx.x] = r;
}
