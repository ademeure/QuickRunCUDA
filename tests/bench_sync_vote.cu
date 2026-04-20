// __syncthreads_count/and/or - fused barrier+vote
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v = (unsigned)threadIdx.x + (unsigned)u2;
    int extra = 0;

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    for (int i = 0; i < ITERS; i++) {
        v = v * 31u + (unsigned)i;
#if MODE == 0
        __syncthreads();
        extra += 1;
#elif MODE == 1
        // Counting barrier
        extra += __syncthreads_count(v & 1);
#elif MODE == 2
        // AND barrier
        extra += __syncthreads_and(v != 0);
#elif MODE == 3
        // OR barrier
        extra += __syncthreads_or(v == (unsigned)seed);
#endif
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v + extra;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1 - t0, (double)(t1-t0)/(double)ITERS);
    }
}
