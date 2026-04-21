// V5 K2: Producer-consumer via DSMEM
// Cluster of 2 CTAs: rank 0 produces data, rank 1 consumes
// Measure end-to-end round-trip via cluster mbarrier
extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[256];
    __shared__ __align__(8) unsigned long long bar;
    unsigned int bar_addr = __cvta_generic_to_shared(&bar);
    unsigned int smem_addr = __cvta_generic_to_shared(smem);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 32;" :: "r"(bar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }

    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait.aligned;");

    unsigned int rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

    unsigned long long t0, t1;
    if (threadIdx.x == 0 && rank == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned int v = (unsigned)u2;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        if (rank == 0) {
            // Producer: write to local SMEM
            smem[threadIdx.x] = (unsigned)i + (unsigned)threadIdx.x;
        } else {
            // Consumer: read from peer (rank 0) SMEM via DSMEM
            unsigned int peer_smem;
            asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                : "=r"(peer_smem) : "r"(smem_addr), "r"(0u));
            unsigned int x;
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                : "=r"(x) : "r"(peer_smem + threadIdx.x * 4));
            v ^= x;
        }
        // Cluster barrier to sync between iterations
        asm volatile("barrier.cluster.arrive;");
        asm volatile("barrier.cluster.wait.aligned;");
    }

    if (threadIdx.x == 0 && rank == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        printf("DSMEM producer-consumer (cluster=2): clk=%llu cy/iter=%.2f\n",
               t1-t0, (double)(t1-t0)/(double)ITERS);
    }
    if (rank == 1 && v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
}
