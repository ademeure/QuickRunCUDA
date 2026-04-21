// V5 J1: Rounding mode throughput — fma.rn vs rz vs rm vs rp
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
#elif MODE == 1
            asm volatile("fma.rz.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
#elif MODE == 2
            asm volatile("fma.rm.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
#elif MODE == 3
            asm volatile("fma.rp.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)a == seed) C[blockIdx.x] = a;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        const char* mode_name[] = {"rn (round nearest)", "rz (zero)", "rm (-inf)", "rp (+inf)"};
        printf("MODE=%d %s clk=%llu cy/fma=%.3f\n",
               MODE, mode_name[MODE], t1-t0, (double)(t1-t0)/(double)ITERS/16.0);
    }
}
