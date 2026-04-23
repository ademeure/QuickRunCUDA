// Test 2c: IADD self-op (only 2 source slots; out=d, src=d)
#ifndef N_INNER
#define N_INNER 1024
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned int)threadIdx.x + 1u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            asm volatile("add.u32 %0, %0, %0;" : "+r"(v));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)v == seed) C[blockIdx.x] = (float)v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("v2c_iadd_selfop total=%llu clk=%llu cy/op=%.4f\n",
               total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
