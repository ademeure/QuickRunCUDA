// Active mask transition / warp divergence cost.
// Mode 0: no branch (baseline), all threads do same work
// Mode 1: if (true) — predictable always-taken, no divergence
// Mode 2: if (threadIdx.x < 16) {...} else {...} — divergence with reconverge
// Mode 3: if (lv & 1) {...} — data-dependent unpredictable divergence
// Mode 4: pure predication via @p add (no branch at all)

#ifndef ITERS_INNER
#define ITERS_INNER 8
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)(threadIdx.x * 131);
    unsigned int b = (unsigned)(threadIdx.x * 271 + 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < ITERS_INNER; k++) {
#if MODE == 0
            v = v * 31u + b + i;
#elif MODE == 1
            // Always-true branch
            if (i + (unsigned)u2 != 0xFFFFFFFFu) {
                v = v * 31u + b + i;
            }
#elif MODE == 2
            // Half-warp divergence
            if (threadIdx.x < 16) {
                v = v * 31u + b + i;
            } else {
                v = v * 17u + b + i;
            }
#elif MODE == 3
            // Data-dependent divergence (random per lane per iter)
            if (v & 1u) {
                v = v * 31u + b + i;
            } else {
                v = v * 17u + b + i;
            }
#elif MODE == 4
            // Predicated (no branch) — compiler should emit @p add
            unsigned int x;
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %3, 0; "
                         "@p mul.lo.u32 %0, %1, 31; @!p mul.lo.u32 %0, %1, 17; "
                         "add.u32 %0, %0, %2; }"
                         : "=r"(x) : "r"(v), "r"(b + i), "r"(v & 1u));
            v = x;
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)ITERS * ITERS_INNER;
        printf("MODE=%d clk=%llu cy/op=%.3f\n",
               MODE, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
