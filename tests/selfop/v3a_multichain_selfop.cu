// Test 3a: 8 INDEPENDENT chains, each chain is "fma a, a, a, a" self-op
// Each chain has its own register; 8 chains run in parallel
#ifndef N_INNER
#define N_INNER 128
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif
#ifndef NCHAINS
#define NCHAINS 8
#endif

#define SO(i) asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(v[i]));

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    float v[NCHAINS];
    #pragma unroll
    for (int k = 0; k < NCHAINS; k++) v[k] = (float)threadIdx.x * 0.001f + 1.001f + (float)k * 1e-6f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            #pragma unroll
            for (int k = 0; k < NCHAINS; k++) {
                asm volatile("fma.rn.f32 %0, %0, %0, %0;" : "+f"(v[k]));
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
        printf("v3a_multi_selfop NCHAINS=%d total=%llu clk=%llu cy/op=%.4f thru=%.3f op/cy\n",
               NCHAINS, total, t1 - t0, (double)(t1-t0)/(double)total,
               (double)total/(double)(t1-t0));
    }
}
