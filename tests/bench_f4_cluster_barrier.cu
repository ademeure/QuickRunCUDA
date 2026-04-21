// F4: Cluster barrier with subset of CTAs
// Cluster size 4 — measure cluster.sync() cost when 4/4 vs 2/4 CTAs present
// Note: cluster.sync requires ALL CTAs in the cluster — can't subset
// What we CAN test: cluster size 1 vs 2 vs 4 vs 8 (varying participation)

#ifndef CSIZE
#define CSIZE 4
#endif

extern "C" __global__ __launch_bounds__(128, 1) __cluster_dims__(CSIZE, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long t0, t1;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Cluster barrier
        asm volatile("barrier.cluster.arrive;");
        asm volatile("barrier.cluster.wait.aligned;");
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("CSIZE=%d clk=%llu cy/sync=%.3f\n",
               CSIZE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
