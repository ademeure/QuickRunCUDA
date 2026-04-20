// FFMA chain latency: 1 warp, 1000 dependent FFMAs in non-unrolled loop, 100 iters.
// Each FFMA must wait for prev result -> measures pure pipeline latency.

#ifndef N_INNER
#define N_INNER 1000
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float v = (float)threadIdx.x + 1.0f;
    float b = 1.0000001f;
    float c = 0.0f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(b), "f"(c));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)v == seed) C[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("N_INNER=%d N_OUTER=%d total_FFMAs=%llu clk=%llu cy/FFMA=%.4f\n",
               N_INNER, N_OUTER, total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
