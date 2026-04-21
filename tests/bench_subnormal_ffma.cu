// Subnormal / denormal FFMA throughput
// Mode 0: normal range (1.0 * 1.0 + 0.5)
// Mode 1: subnormal inputs (1e-40)
// Mode 2: subnormal output (mul to flush range)
// Mode 3: ftz forced via PTX hint
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float x, y, z;
#if MODE == 0
    x = 1.000001f + (float)u2 * 1e-9f;
    y = 1.000002f + (float)u2 * 1e-9f;
    z = 0.5f;
#elif MODE == 1
    // subnormal inputs via bit-cast (0x00000100 ~= 3.6e-43, subnormal)
    unsigned int xi=0x00000100u^(unsigned)u2, yi=0x00000200u^(unsigned)u2, zi=0x00000300u;
    x = __int_as_float(xi); y = __int_as_float(yi); z = __int_as_float(zi);
#elif MODE == 2
    // produce subnormal output: small * small underflows
    unsigned int xi=0x18000001u^(unsigned)u2, yi=0x18000002u^(unsigned)u2, zi=0x00000001u;
    x = __int_as_float(xi); y = __int_as_float(yi); z = __int_as_float(zi);
#elif MODE == 3
    // subnormal input + FTZ form
    unsigned int xi=0x00000100u^(unsigned)u2, yi=0x00000200u^(unsigned)u2, zi=0x00000300u;
    x = __int_as_float(xi); y = __int_as_float(yi); z = __int_as_float(zi);
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // 4 chained FFMA
#if MODE == 3
        asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
        asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
        asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
        asm volatile("fma.rn.ftz.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
#else
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
        asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(x) : "f"(y), "f"(z));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)(x*1e30f) == seed) C[blockIdx.x] = x;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.2f cy/fma=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/4.0);
    }
}
