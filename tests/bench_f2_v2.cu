// F2 v2: all 32 lanes always run __syncwarp(MASK) — no divergence
// Note: __syncwarp(mask) requires all named threads in mask to converge
// If mask doesn't include all 32 threads, behavior is undefined for excluded ones
// Here we test all 32 lanes calling syncwarp with various masks
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int x = 0xDEADBEEFu ^ (unsigned)u2;

    unsigned int mask;
#if MODE == 0
    // 0xFFFFFFFF compile-time const
    mask = 0xFFFFFFFFu;
#elif MODE == 1
    // 0xFFFFFFFF runtime (defeat compile-time)
    mask = (u2 == -999) ? 0u : 0xFFFFFFFFu;
#elif MODE == 2
    // PTX bar.warp.sync directly with constant
    mask = 0xFFFFFFFFu;
#elif MODE == 3
    // PTX bar.warp.sync with runtime full mask
    mask = (u2 == -999) ? 0u : 0xFFFFFFFFu;
#elif MODE == 4
    // bar.sync 0 (CTA-wide) — 1 warp = same as warp barrier
    mask = 0xFFFFFFFFu;
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
            v ^= x;
#if MODE == 0
            __syncwarp(0xFFFFFFFFu);
#elif MODE == 1
            __syncwarp(mask);
#elif MODE == 2
            asm volatile("bar.warp.sync 0xFFFFFFFF;");
#elif MODE == 3
            asm volatile("bar.warp.sync %0;" :: "r"(mask));
#elif MODE == 4
            asm volatile("bar.sync 0;");
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * 32 + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f cy/sync=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/16.0);
    }
}
