// Same-pipe ILP test: can 2 independent FFMAs issue in 1 cycle?
// If issue port = 1/SMSP/cy, even fully independent FFMAs serialize at issue.
// Test by varying number of INDEPENDENT chains within one warp.

#ifndef N_CHAINS
#define N_CHAINS 1
#endif
#ifndef N_INNER
#define N_INNER 100
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float v[16];
    float fb = 1.0000001f + (float)u2 * 1e-9f;
    #pragma unroll
    for (int k = 0; k < 16; k++) v[k] = (float)(threadIdx.x + k);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            // N_CHAINS independent FFMA streams (each chain dep through v[k])
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                asm volatile("fma.rn.f32 %0, %0, %1, %0;"
                             : "+f"(v[k]) : "f"(fb));
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float acc = 0;
    #pragma unroll
    for (int k = 0; k < 16; k++) acc += v[k];
    if ((int)acc == seed) C[blockIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long ffmas = (unsigned long long)N_OUTER * N_INNER * N_CHAINS;
        printf("N_CHAINS=%d N_INNER=%d ffmas=%llu clk=%llu cy/FFMA=%.4f cy/iter=%.4f\n",
               N_CHAINS, N_INNER, ffmas, t1 - t0,
               (double)(t1-t0)/(double)ffmas,
               (double)(t1-t0)/(double)(N_OUTER*N_INNER));
    }
}
