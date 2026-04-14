// Cluster (CGA) barriers and distributed-shared memory on sm_100+.
// Requires __cluster_dims__ and cooperative-launch.

#ifndef OP
#define OP 0
#endif
#ifndef UNROLL
#define UNROLL 16
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __cluster_dims__(2, 1, 1) __launch_bounds__(128, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x < 64) smem[threadIdx.x] = threadIdx.x;

    unsigned int v = threadIdx.x;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // barrier.cluster.arrive; barrier.cluster.wait
            asm volatile("barrier.cluster.arrive;");
            asm volatile("barrier.cluster.wait;");
#elif OP == 1  // barrier.cluster.arrive.relaxed
            asm volatile("barrier.cluster.arrive.relaxed;");
            asm volatile("barrier.cluster.wait;");
#elif OP == 2  // barrier.cluster.sync
            asm volatile("barrier.cluster.sync;");
#elif OP == 3  // mbarrier test — needs init, skip here
            v = v + 1;
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
