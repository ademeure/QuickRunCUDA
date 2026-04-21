// D6: RF port pressure — FFMA with N unique sources per inst
// MODE 0: a*a + a (1 unique source)
// MODE 1: a*b + a (2 unique)
// MODE 2: a*b + c (3 unique — full FMA dataflow)
// MODE 3: 4 unique sources rotated each inst (a*b+c, c*d+a, ...)
// MODE 4: same as 3 but with .reuse hint
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f + (float)u2 * 1e-9f;
    float d = 0.7f + (float)u2 * 2e-9f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // a = a*a + a (1 unique read source)
            asm("fma.rn.f32 %0, %0, %0, %0;" : "+f"(a));
#elif MODE == 1
            // a = a*b + a (2 unique sources)
            asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(a) : "f"(b));
#elif MODE == 2
            // a = a*b + c (3 unique sources, classic FMA)
            asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
#elif MODE == 3
            // 4 unique sources rotated; each FMA reads different combo
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a) : "f"(a), "f"(b), "f"(c));
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(b) : "f"(b), "f"(c), "f"(d));
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(c) : "f"(c), "f"(d), "f"(a));
            asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(d) : "f"(d), "f"(a), "f"(b));
#elif MODE == 4
            // 4 indep accumulators with shared y, z (broadcast — should hit .reuse)
            asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
            asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(d) : "f"(b), "f"(c));
            float e = a + b;
            asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(e) : "f"(b), "f"(c));
            float f = c + d;
            asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(f) : "f"(b), "f"(c));
            a = e + f;
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sink = a + b + c + d;
    if ((int)sink == seed) C[blockIdx.x] = sink;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
#if MODE == 3
        int ops = 4;
#elif MODE == 4
        int ops = 4;
#else
        int ops = 1;
#endif
        printf("MODE=%d clk=%llu cy/iter=%.3f cy/fma=%.4f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/16.0/(double)ops);
    }
}
