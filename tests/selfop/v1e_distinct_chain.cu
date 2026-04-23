// Test 1e: Distinct single chain
// Output != any source. Each iteration consumes prev result via *one* operand
// fma.rn.f32 d_new, d_prev, k1, k2  (chain via mul1, all 3 operands distinct regs)
// We rotate: out goes to a fresh reg; then becomes the "d_prev" of next iter
// Cleanest expression: ping-pong two accumulators
#ifndef N_INNER
#define N_INNER 1024
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    float v0 = (float)threadIdx.x * 0.001f + 1.0001f;
    float v1 = 0.0f;
    float k1 = 1.000001f + (float)u1 * 1e-9f;
    float k2 = 1e-7f + (float)u2 * 1e-9f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Inner unroll = 2 (ping-pong). Outer is ITERS / 2.
    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll (N_INNER/2)
        for (int j = 0; j < (N_INNER/2); j++) {
            // v1 = v0 * k1 + k2;
            // v0 = v1 * k1 + k2;
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(v1) : "f"(v0), "f"(k1), "f"(k2));
            asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(v0) : "f"(v1), "f"(k1), "f"(k2));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float v = v0 + v1;
    if ((int)v == seed) C[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("v1e_distinct_chain total=%llu clk=%llu cy/op=%.4f\n",
               total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
