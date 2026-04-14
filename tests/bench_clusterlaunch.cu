// Distributed shared memory (DSM) in cluster — writing to other block's SMEM.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __cluster_dims__(2,1,1) __launch_bounds__(128, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x < 128) smem[threadIdx.x] = threadIdx.x + blockIdx.x * 1000;
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");

    unsigned int my_addr = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x]);
    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // Standard LDS (same block)
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(my_addr));
            v ^= r;
#elif OP == 1  // mapa: map shared addr to peer block in cluster
            unsigned int mapped;
            unsigned int target_rank = 1 - (blockIdx.x & 1);  // pair up with neighbor
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(mapped) : "r"(my_addr), "r"(target_rank));
            unsigned int r;
            asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r) : "r"(mapped));
            v ^= r;
#elif OP == 2  // cluster-shared atomic add
            unsigned int mapped;
            unsigned int target_rank = 1 - (blockIdx.x & 1);
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(mapped) : "r"(my_addr), "r"(target_rank));
            unsigned int r;
            asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;"
                : "=r"(r) : "r"(mapped), "r"(1u));
            v ^= r;
#endif
        }
    }
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait;");
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
