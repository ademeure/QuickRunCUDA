// shfl with width=1 (effectively nop — each lane is its own group)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = (unsigned)threadIdx.x;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
        v = v * 31u + (unsigned)i;
#if MODE == 0
        // baseline
#elif MODE == 1
        // shfl width=1 (degenerate — each lane is its own group)
        v = __shfl_sync(0xFFFFFFFFu, v, 0, 1);
#elif MODE == 2
        // shfl width=32 (full warp)
        v = __shfl_sync(0xFFFFFFFFu, v, 0, 32);
#elif MODE == 3
        // shfl_xor width=1
        v = __shfl_xor_sync(0xFFFFFFFFu, v, 1, 1);
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
