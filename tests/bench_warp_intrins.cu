// Warp value matching + lane election
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)threadIdx.x | (unsigned)(u2 << 16);
    unsigned int acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        v = v * 31u + (unsigned)i;
#if MODE == 0
        // Baseline: just compute
        acc ^= v;
#elif MODE == 1
        // __match_any_sync: find lanes with same value
        unsigned int mask = __match_any_sync(0xFFFFFFFFu, v & 0x3);
        acc ^= mask;
#elif MODE == 2
        // __match_all_sync: all-equal vote
        int pred;
        unsigned int mask = __match_all_sync(0xFFFFFFFFu, v & 0x3, &pred);
        acc ^= mask + (unsigned)pred;
#elif MODE == 3
        // PTX elect.sync (one lane wins)
        unsigned int leader;
        asm("{ .reg .pred p; elect.sync %0|p, 0xFFFFFFFF; selp.b32 %0, 1, 0, p; }"
            : "=r"(leader));
        acc ^= leader;
#elif MODE == 4
        // PTX vote.uni - uniform check
        unsigned int uni;
        asm("{ .reg .pred p; vote.sync.uni.pred p, 0x1, 0xFFFFFFFF; selp.b32 %0, 1, 0, p; }"
            : "=r"(uni));
        acc ^= uni;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
