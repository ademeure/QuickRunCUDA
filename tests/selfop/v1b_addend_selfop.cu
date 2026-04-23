// Test 1b: addend self-op (chain via addend slot only)
// fma.rn.f32 %0, %1, %2, %0  -- mul1, mul2 are distinct constants; addend = chain
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
    float k1 = 0.5f + (float)u1 * 1e-9f;   // runtime input -> not folded
    float k2 = 0.25f + (float)u2 * 1e-9f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            // d = k1 * k2 + d
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(v) : "f"(k1), "f"(k2));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)v == seed) C[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
        printf("v1b_addend_selfop total=%llu clk=%llu cy/op=%.4f\n",
               total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
