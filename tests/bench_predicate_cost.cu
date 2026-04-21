// Predicated FFMA cost: always-true, always-false, mixed
// Measure if @P adds latency when predicate is uniform
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    float x = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float y = (float)(threadIdx.x ^ u2) * 0.002f + 1.0f;
    float z = 0.5f;

    // Build predicate based on u2 — runtime so compiler can't fold
    unsigned int p_input;
#if MODE == 0
    // baseline: 4 FFMA, no predicate
    p_input = 1;
#elif MODE == 1
    // always-true predicate (uniform across warp)
    p_input = (u2 != -999) ? 1 : 0;
#elif MODE == 2
    // always-false predicate (uniform across warp)
    p_input = (u2 == -999) ? 1 : 0;
#elif MODE == 3
    // half-warp split predicate
    p_input = (threadIdx.x < 16) ? 1 : 0;
#elif MODE == 4
    // random per-lane (data-dependent)
    p_input = (threadIdx.x ^ u2) & 1;
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        x = x*y + z; x = x*y + z; x = x*y + z; x = x*y + z;
#else
        // 4 FFMAs guarded by @p
        asm volatile("{ .reg .pred q;\n"
                     "  setp.ne.b32 q, %3, 0;\n"
                     "  @q fma.rn.f32 %0, %0, %1, %2;\n"
                     "  @q fma.rn.f32 %0, %0, %1, %2;\n"
                     "  @q fma.rn.f32 %0, %0, %1, %2;\n"
                     "  @q fma.rn.f32 %0, %0, %1, %2; }"
                     : "+f"(x) : "f"(y), "f"(z), "r"(p_input));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)x == seed) C[blockIdx.x] = x;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.2f cy/fma=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS/4.0);
    }
}
