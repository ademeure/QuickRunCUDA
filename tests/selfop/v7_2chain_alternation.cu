// Test 7: 2-chain alternation
// Chain A and B alternate; each new B reads what A wrote 1 cy ago
// Designed to expose back-to-back dependent issue
#ifndef N_INNER
#define N_INNER 1024
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    float a = (float)threadIdx.x * 0.001f + 1.0001f;
    float b = a + 0.5f;
    float k1 = 1.000001f + (float)u1 * 1e-9f;
    float k2 = 1e-7f + (float)u2 * 1e-9f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Pattern:
    //   b = a*k1 + k2;   <- b depends on a
    //   a = b*k1 + k2;   <- a depends on b
    // True dependency every step.  This is identical to a single chain in
    // terms of dependence depth, but the destination ALTERNATES regs.
    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll (N_INNER/2)
        for (int j = 0; j < (N_INNER/2); j++) {
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(b) : "f"(a), "f"(k1), "f"(k2));
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(a) : "f"(b), "f"(k1), "f"(k2));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float v = a + b;
    if ((int)v == seed) C[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("v7_2chain_alternation total=%llu clk=%llu cy/op=%.4f\n",
               total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
