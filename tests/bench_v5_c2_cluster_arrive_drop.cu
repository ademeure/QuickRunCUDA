// V5 C2: cluster.barrier cost vs cluster size (CSIZE = 2, 4, 6, 8)
// Goal: characterize how cluster.barrier latency scales with # CTAs
// MODE controls cluster size; MODE 0..3 = 2, 4, 6, 8 CTAs
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define CSIZE 2
#elif MODE == 1
#define CSIZE 4
#elif MODE == 2
#define CSIZE 6
#elif MODE == 3
#define CSIZE 8
#endif

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(CSIZE, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d CSIZE=%d cy/iter=%.3f\n",
               MODE, CSIZE, (double)(t1-t0)/(double)ITERS);
    }
}
