// Branch overlap with compute: can BRA hide behind FFMA?
// Mode 0: pure FFMA loop
// Mode 1: FFMA + always-taken predictable branch (BRA U)
// Mode 2: FFMA + data-dep unpredictable branch (50/50)
// Mode 3: FFMA + always-not-taken predictable

#ifndef MODE
#define MODE 0
#endif
#ifndef N_INNER
#define N_INNER 100
#endif
#ifndef N_OUTER
#define N_OUTER 1000
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float fv = (float)threadIdx.x + 1.5f;
    float fb = 1.0000001f + (float)u2 * 1e-9f;
    float fc = 0.0000001f;
    int extra = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#if MODE == 1
            // Always-taken (i+u2 always non-zero in our setup)
            if ((i + (unsigned)u2) != 0xFFFFFFFFu) {
                extra++;
            }
#elif MODE == 2
            // Data-dependent unpredictable
            if (((unsigned)__float_as_uint(fv) & 1) == 0) {
                extra++;
            }
#elif MODE == 3
            // Always-not-taken
            if ((i + (unsigned)u2) == 0xFFFFFFFFu) {
                extra++;
            }
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)fv == seed && extra == seed) C[blockIdx.x] = fv + (float)extra;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * N_INNER;
        printf("MODE=%d ffmas=%llu clk=%llu cy/iter=%.4f\n",
               MODE, total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
