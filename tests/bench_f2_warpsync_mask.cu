// F2: __syncwarp() with various masks — does mask choice cost more?
// MODE 0: 0xFFFFFFFF (full warp — most common)
// MODE 1: 0x0000FFFF (lower half)
// MODE 2: 0xFFFF0000 (upper half)
// MODE 3: 0xAAAAAAAA (alternating)
// MODE 4: 0x55555555 (alternating opposite)
// MODE 5: 0x00000001 (single lane)
// MODE 6: runtime-computed mask (defeat const-fold)
// MODE 7: no sync (baseline)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int x = 0xDEADBEEFu ^ (unsigned)u2;

    // Per-mode mask (some are runtime-derived to defeat folding)
    unsigned int mask;
#if MODE == 0
    mask = 0xFFFFFFFFu;
#elif MODE == 1
    mask = 0x0000FFFFu;
#elif MODE == 2
    mask = 0xFFFF0000u;
#elif MODE == 3
    mask = 0xAAAAAAAAu;
#elif MODE == 4
    mask = 0x55555555u;
#elif MODE == 5
    mask = 0x00000001u;
#elif MODE == 6
    mask = (u2 == -999) ? 0x12345678u : 0xFFFFFFFFu;
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
            v ^= x;
#if MODE != 7
            // Only the threads in mask participate; others must skip the bar
            if (mask & (1u << threadIdx.x)) {
                __syncwarp(mask);
            }
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
