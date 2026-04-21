// V10: FFMA 2-source vs 3-source throughput
// V6 D6 claimed 3-source hits only 65% due to RF 2 read ports/cy.
// Verify with full ncu pipe_fma measurement.
#ifndef SOURCES
#define SOURCES 2   // 2=self-feed, 3=three distinct sources
#endif
#ifndef N_CHAINS
#define N_CHAINS 8
#endif

extern "C" __global__ __launch_bounds__(256, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float v[N_CHAINS], b[N_CHAINS], c[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) {
        v[k] = (float)(threadIdx.x + k) * 0.001f;
        b[k] = (float)(threadIdx.x * 2 + k) * 0.001f;
        c[k] = (float)(threadIdx.x * 3 + k) * 0.001f;
    }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += 16) {
        #pragma unroll
        for (int j = 0; j < 16; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if SOURCES == 2
                // v = v * b + v (2 distinct sources: v, b)
                asm volatile("fma.rn.f32 %0, %0, %1, %0;" : "+f"(v[k]) : "f"(b[k]));
#elif SOURCES == 3
                // v = v * b + c (3 distinct sources: v, b, c)
                asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(v[k]) : "f"(b[k]), "f"(c[k]));
#endif
            }
        }
    }

    float acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc += v[k];
    if ((int)acc == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
