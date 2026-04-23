// Test 3b: 8 INDEPENDENT chains, each chain uses distinct sources
// fma d, d_prev, k1, k2 — chain via mul1 only
#ifndef N_INNER
#define N_INNER 128
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif
#ifndef NCHAINS
#define NCHAINS 8
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    float v[NCHAINS];
    #pragma unroll
    for (int k = 0; k < NCHAINS; k++) v[k] = (float)threadIdx.x * 0.001f + 1.001f + (float)k * 1e-6f;
    float k1 = 1.000001f + (float)u1 * 1e-9f;
    float k2 = 1e-7f + (float)u2 * 1e-9f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            #pragma unroll
            for (int k = 0; k < NCHAINS; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v[k]) : "f"(k1), "f"(k2));
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float vsum = 0;
    #pragma unroll
    for (int k = 0; k < NCHAINS; k++) vsum += v[k];
    if ((int)vsum == seed) C[blockIdx.x] = vsum;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * N_INNER * NCHAINS;
        printf("v3b_multi_distinct NCHAINS=%d total=%llu clk=%llu cy/op=%.4f thru=%.3f op/cy\n",
               NCHAINS, total, t1 - t0, (double)(t1-t0)/(double)total,
               (double)total/(double)(t1-t0));
    }
}
