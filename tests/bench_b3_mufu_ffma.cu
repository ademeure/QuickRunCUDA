// B3: MUFU + FFMA overlap — explicit measurement
// MUFU is XU pipe, FFMA is FMA pipe — should fully overlap
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)(threadIdx.x ^ u2) * 0.001f + 0.5f;
    float b = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float c = 0.5f + (float)u2 * 1e-9f;
    float m = (float)(threadIdx.x ^ u2) * 0.5f + 1.0f;  // MUFU input

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // MUFU only (rsqrt chain)
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
#elif MODE == 1
            // FFMA only (chained, single chain)
            a = a*b + c;
#elif MODE == 2
            // 1 MUFU + 1 FFMA per iter
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            a = a*b + c;
#elif MODE == 3
            // 1 MUFU + 4 FFMA — likely fits MUFU window
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            a = a*b + c; a = a*b + c; a = a*b + c; a = a*b + c;
#elif MODE == 4
            // 1 MUFU + 8 FFMA
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            a = a*b + c; a = a*b + c; a = a*b + c; a = a*b + c;
            a = a*b + c; a = a*b + c; a = a*b + c; a = a*b + c;
#elif MODE == 5
            // 4 MUFU only
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
#elif MODE == 6
            // 4 MUFU + 4 FFMA
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(m));
            a = a*b + c; a = a*b + c; a = a*b + c; a = a*b + c;
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sink = a + m;
    if ((int)sink == seed) C[blockIdx.x] = sink;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
