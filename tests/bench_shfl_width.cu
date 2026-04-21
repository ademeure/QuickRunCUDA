// SHFL with width parameter (sub-warp partition)
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
        // No shfl
        v = v * 31u + (unsigned)i;
#elif MODE == 1
        // Full warp shfl_xor (width=32)
        v = v * 31u + (unsigned)i;
        v = __shfl_xor_sync(0xFFFFFFFFu, v, 1, 32);
#elif MODE == 2
        // Half warp shfl_xor (width=16)
        v = v * 31u + (unsigned)i;
        v = __shfl_xor_sync(0xFFFFFFFFu, v, 1, 16);
#elif MODE == 3
        // Quarter warp (width=8)
        v = v * 31u + (unsigned)i;
        v = __shfl_xor_sync(0xFFFFFFFFu, v, 1, 8);
#elif MODE == 4
        // Width=4
        v = v * 31u + (unsigned)i;
        v = __shfl_xor_sync(0xFFFFFFFFu, v, 1, 4);
#elif MODE == 5
        // Width=2
        v = v * 31u + (unsigned)i;
        v = __shfl_xor_sync(0xFFFFFFFFu, v, 1, 2);
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
