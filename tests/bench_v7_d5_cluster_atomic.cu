// V7 D5: Cluster-wide atomic (atom.shared::cluster) vs per-CTA atomics
// MODE 0: each CTA atomic on its own counter
// MODE 1: all CTAs atomic on CTA 0's counter via cluster
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(4, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int counter;
    if (threadIdx.x == 0) counter = 0;

    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    unsigned int local_addr = __cvta_generic_to_shared(&counter);
    unsigned int target_addr;
#if MODE == 0
    target_addr = local_addr;  // own SMEM
#elif MODE == 1
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;" : "=r"(target_addr) : "r"(local_addr));
#endif

    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        unsigned int old;
        asm volatile("atom.shared::cta.add.u32 %0, [%1], %2;"
                     : "=r"(old) : "r"(target_addr), "r"((unsigned)threadIdx.x));
#elif MODE == 1
        unsigned int old;
        asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;"
                     : "=r"(old) : "r"(target_addr), "r"((unsigned)threadIdx.x));
#endif
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d cy/iter=%.3f\n", MODE, (double)(t1-t0)/(double)ITERS);
    }
}
