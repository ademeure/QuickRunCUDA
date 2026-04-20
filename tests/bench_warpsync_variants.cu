// __syncwarp(mask) vs bar.warp.sync mask SASS + perf comparison.

#ifndef MODE
#define MODE 0
#endif
#ifndef N_ITERS
#define N_ITERS 100000
#endif

extern "C" __global__ __launch_bounds__(32, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_ITERS; i++) {
        v = v * 31u + (unsigned)i;
#if MODE == 0
        // No sync (baseline)
#elif MODE == 1
        __syncwarp(0xFFFFFFFFu);
#elif MODE == 2
        __syncwarp(0x0000FFFFu);  // 16 lanes
#elif MODE == 3
        asm volatile("bar.warp.sync 0xFFFFFFFF;");
#elif MODE == 4
        asm volatile("bar.warp.sync 0x0000FFFF;");
#elif MODE == 5
        asm volatile("bar.warp.sync %0;" :: "r"((unsigned)u2 | 0xFFFFFFFFu));  // dynamic mask
#endif
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d N_ITERS=%d clk=%llu cy/iter=%.3f\n",
               MODE, N_ITERS, t1 - t0, (double)(t1-t0)/(double)N_ITERS);
    }
}
