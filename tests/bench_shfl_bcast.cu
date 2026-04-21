// SHFL broadcast cost — fundamental warp-comm pattern
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // baseline
        v = v * 31u + (unsigned)i;
#elif MODE == 1
        // shfl broadcast from lane 0 (constant)
        v = v * 31u + (unsigned)i;
        v = __shfl_sync(0xFFFFFFFFu, v, 0);
#elif MODE == 2
        // shfl broadcast from variable lane
        v = v * 31u + (unsigned)i;
        v = __shfl_sync(0xFFFFFFFFu, v, ((unsigned)i + (unsigned)u2) & 31);
#elif MODE == 3
        // shfl broadcast that's compiler-folded constant
        v = v * 31u + (unsigned)i;
        v = __shfl_sync(0xFFFFFFFFu, v, threadIdx.x);  // identity (no broadcast)
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
