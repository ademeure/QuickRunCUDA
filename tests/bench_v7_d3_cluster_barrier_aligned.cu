// V7 D3: cluster.barrier vs cluster.barrier.aligned
// MODE 0: with .aligned (assumes all threads call same instruction)
// MODE 1: without .aligned
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(4, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
#else
        asm volatile("barrier.cluster.arrive;");
        asm volatile("barrier.cluster.wait;");
#endif
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d cy/iter=%.3f\n", MODE, (double)(t1-t0)/(double)ITERS);
    }
}
