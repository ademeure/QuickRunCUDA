// D6 v4: defeat .reuse — every FFMA reads 3 UNIQUE registers, no broadcast
// Use distinct za_k for each chain
#ifndef NCHAINS
#define NCHAINS 16
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float a[16];
    float ya[16], za[16];  // each chain has its own ya AND za — no broadcast
    for (int k = 0; k < NCHAINS; k++) {
        a[k] = (float)(threadIdx.x ^ u2) * 0.001f * (k+1);
        ya[k] = (float)(threadIdx.x ^ (u2+k)) * 0.002f + 1.0f;
        za[k] = (float)(threadIdx.x ^ (u2+k+100)) * 0.003f + 0.5f;
    }

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
            #pragma unroll
            for (int k = 0; k < NCHAINS; k++) {
                asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a[k]) : "f"(ya[k]), "f"(za[k]));
            }
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sink = 0;
    for (int k = 0; k < NCHAINS; k++) sink += a[k];
    if ((int)sink == seed) C[blockIdx.x] = sink;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total_fma = (unsigned long long)ITERS * 16 * NCHAINS;
        printf("NCHAINS=%d clk=%llu cy/fma=%.4f thru=%.3f fma/cy\n",
               NCHAINS, t1-t0,
               (double)(t1-t0)/(double)total_fma,
               (double)total_fma / (double)(t1-t0));
    }
}
