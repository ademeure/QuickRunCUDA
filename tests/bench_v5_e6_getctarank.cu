// V5 E6: getctarank semantics
// Test PTX %cluster_ctarank in cluster vs non-cluster contexts
// And the PTX getctarank instruction (cluster-only)
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
extern "C" __global__ __launch_bounds__(32, 1)
#elif MODE == 1
extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(4, 1, 1)
#endif
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    unsigned int rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

    unsigned int nrank;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(nrank));

    unsigned int cluster_id_x;
    asm volatile("mov.u32 %0, %%clusterid.x;" : "=r"(cluster_id_x));

    unsigned int cta_in_cluster_x;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(cta_in_cluster_x));

    if (blockIdx.x < 8) {
        printf("MODE=%d block=%d: rank=%u nctas=%u clusterid.x=%u cta_in_cluster.x=%u\n",
               MODE, blockIdx.x, rank, nrank, cluster_id_x, cta_in_cluster_x);
    }
    ((unsigned int*)C)[blockIdx.x] = rank;
}
