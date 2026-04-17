// Test if NOP after S2UR can be hidden by inter-clock FFMA work.
// Compare: 2 clocks back-to-back vs 2 clocks with N FFMAs between.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef N_FMA
#define N_FMA 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    float v0 = __int_as_float(tid+1)*1e-30f;
    float v1 = __int_as_float(tid+2)*1e-30f;
    float v2 = __int_as_float(tid+3)*1e-30f;
    float v3 = __int_as_float(tid+4)*1e-30f;
    float y = 1.5f;

    unsigned t0, t1;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
    #pragma unroll
    for (int k = 0; k < N_FMA; k++) {
        v0 = v0 * y + v0;
        v1 = v1 * y + v1;
        v2 = v2 * y + v2;
        v3 = v3 * y + v3;
    }
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));

    if (tid == 0) {
        ((unsigned*)C)[blockIdx.x] = t1 - t0;
        // Force compute
        float sum = v0+v1+v2+v3;
        if (__float_as_int(sum) == seed) C[blockIdx.x + 1024] = sum;
    }
}
