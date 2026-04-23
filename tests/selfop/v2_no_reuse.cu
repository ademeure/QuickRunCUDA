// Test 2-i / 2-ii: same SASS pattern with/without .reuse opportunity
// 2-i: chain via mul1 with a single shared constant -> compiler will add .reuse
// (this is essentially v1c)
// 2-ii: chain via mul1 where the constant *changes every iteration*
//      -> no .reuse opportunity. We use a shared array and rotate index.
// Comparing 2-i and 2-ii isolates: does losing .reuse change cy/op?
#ifndef N_INNER
#define N_INNER 1024
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    float v = (float)threadIdx.x * 0.001f + 1.0001f;

    // 16 distinct constants in regs -> compiler can't .reuse the same source
    float c0  = 1.000001f + (float)u1 * 1e-9f;
    float c1  = 1.000002f + (float)u1 * 1e-9f;
    float c2  = 1.000003f + (float)u1 * 1e-9f;
    float c3  = 1.000004f + (float)u1 * 1e-9f;
    float c4  = 1.000005f + (float)u1 * 1e-9f;
    float c5  = 1.000006f + (float)u1 * 1e-9f;
    float c6  = 1.000007f + (float)u1 * 1e-9f;
    float c7  = 1.000008f + (float)u1 * 1e-9f;
    float k2 = 1e-7f + (float)u2 * 1e-9f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        // Block of 8 FFMAs, each with a DIFFERENT constant in mul1 slot
        // (cycle through c0..c7) -- defeats .reuse for src_a
        #pragma unroll (N_INNER/8)
        for (int j = 0; j < (N_INNER/8); j++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(c0), "f"(k2));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(c1), "f"(k2));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(c2), "f"(k2));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(c3), "f"(k2));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(c4), "f"(k2));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(c5), "f"(k2));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(c6), "f"(k2));
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v) : "f"(c7), "f"(k2));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)v == seed) C[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("v2_no_reuse total=%llu clk=%llu cy/op=%.4f\n",
               total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
