// __activemask cost vs known full mask
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)threadIdx.x;
    unsigned int acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        v = v * 31u + (unsigned)i;
#elif MODE == 1
        // Vote ballot to find active lanes
        unsigned int mask = __activemask();
        v = v * 31u + mask;
#elif MODE == 2
        // PTX vote.ballot
        unsigned int mask;
        asm("vote.sync.ballot.b32 %0, 0x1, 0xFFFFFFFF;" : "=r"(mask));
        v = v * 31u + mask;
#elif MODE == 3
        // PTX __ballot_sync (constant condition)
        unsigned int mask = __ballot_sync(0xFFFFFFFF, 1);
        v = v * 31u + mask;
#elif MODE == 4
        // ANY_SYNC
        int any = __any_sync(0xFFFFFFFF, threadIdx.x > 16);
        v = v * 31u + (unsigned)any;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
