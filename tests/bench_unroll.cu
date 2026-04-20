// Loop unrolling thresholds.
// Mode 0: no pragma (compiler default — usually unrolls if it can prove bound)
// Mode 1: #pragma unroll 1 (force serial)
// Mode 2: #pragma unroll 4
// Mode 3: #pragma unroll (full)
// Mode 4: #pragma unroll 16

#ifndef MODE
#define MODE 0
#endif
#ifndef N_INNER
#define N_INNER 32
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float fv = (float)threadIdx.x + 1.0f;
    float fb = 1.000001f + (float)u2 * 1e-9f;
    float fc = 0.000001f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        for (int j = 0; j < N_INNER; j++)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 1
        #pragma unroll 1
        for (int j = 0; j < N_INNER; j++)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 2
        #pragma unroll 4
        for (int j = 0; j < N_INNER; j++)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 3
        #pragma unroll
        for (int j = 0; j < N_INNER; j++)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#elif MODE == 4
        #pragma unroll 16
        for (int j = 0; j < N_INNER; j++)
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)fv == seed) C[blockIdx.x] = fv;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)ITERS * N_INNER;
        printf("MODE=%d total_FFMAs=%llu clk=%llu cy/FFMA=%.4f\n",
               MODE, total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
