// Branch density: how does N branches per FFMA affect throughput?

#ifndef MODE
#define MODE 0
#endif
#ifndef N_INNER
#define N_INNER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float fv = (float)threadIdx.x + 1.5f;
    float fb = 1.000001f + (float)u2 * 1e-9f;
    float fc = 0.000001f;
    int extra = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#if MODE >= 1
            if ((i + j + (unsigned)u2) != 0xFFFFFFFFu) extra++;
#endif
#if MODE >= 2
            if ((i + j*7 + (unsigned)u2) != 0xFFFFFFFFu) extra++;
#endif
#if MODE >= 4
            if ((i + j*13 + (unsigned)u2) != 0xFFFFFFFFu) extra++;
            if ((i + j*17 + (unsigned)u2) != 0xFFFFFFFFu) extra++;
#endif
#if MODE >= 8
            if ((i + j*19 + (unsigned)u2) != 0xFFFFFFFFu) extra++;
            if ((i + j*23 + (unsigned)u2) != 0xFFFFFFFFu) extra++;
            if ((i + j*29 + (unsigned)u2) != 0xFFFFFFFFu) extra++;
            if ((i + j*31 + (unsigned)u2) != 0xFFFFFFFFu) extra++;
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)fv == seed && extra == seed) C[blockIdx.x] = fv + (float)extra;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)ITERS * N_INNER;
        printf("MODE=%d total_iters=%llu clk=%llu cy/iter=%.3f\n",
               MODE, total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
