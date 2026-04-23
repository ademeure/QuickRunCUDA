// Test 2d: DFMA self-op (FP64)
#ifndef N_INNER
#define N_INNER 256
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    double v = (double)threadIdx.x * 0.001 + 1.0001;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            asm volatile("fma.rn.f64 %0, %0, %0, %0;" : "+d"(v));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)v == seed) C[blockIdx.x] = (float)v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("v2d_dfma_selfop total=%llu clk=%llu cy/op=%.4f\n",
               total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
