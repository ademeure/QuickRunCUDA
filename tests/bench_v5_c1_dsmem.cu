// V5 C1: DSMEM (Distributed SMEM) latency
// In a cluster, each CTA has its own SMEM but can read peer CTA SMEM via DSMEM addressing
// MODE 0: local SMEM read (baseline)
// MODE 1: peer SMEM in same TPC (cluster CTA 1 from CTA 0)
// MODE 2: peer SMEM via cluster CTA 2 or 3 (across GPC row when CSIZE=4)
#ifndef MODE
#define MODE 0
#endif
#ifndef CSIZE
#define CSIZE 4
#endif

extern "C" __global__ __launch_bounds__(32, 1) __cluster_dims__(CSIZE, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[1024];
    if (threadIdx.x == 0) for (int i = 0; i < 1024; i++) smem[i] = i + (unsigned)u2;

    // Cluster sync to ensure all CTAs initialized
    asm volatile("barrier.cluster.arrive;");
    asm volatile("barrier.cluster.wait.aligned;");

    unsigned int rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));

    // Determine target CTA for read
    unsigned int target_cta;
#if MODE == 0
    target_cta = rank;  // local SMEM
#elif MODE == 1
    target_cta = (rank == 0) ? 1 : 0;  // peer (TPC sibling for CSIZE=2)
#elif MODE == 2
    target_cta = (rank ^ 2) % CSIZE;  // across-row CTA when CSIZE=4
#endif

    // Get peer SMEM base via cluster mapping
    unsigned int local_smem_addr = __cvta_generic_to_shared(smem);
    unsigned int peer_smem_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
        : "=r"(peer_smem_addr) : "r"(local_smem_addr), "r"(target_cta));

    unsigned int v = (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int x;
        // Each iter use chained address (defeat constant fold + force chain dep)
        unsigned int off = (v & 1023) * 4;
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(x) : "r"(peer_smem_addr + off));
        v = x;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (rank == 0 && threadIdx.x == 0) {
        printf("MODE=%d CSIZE=%d clk=%llu cy/load=%.2f\n",
               MODE, CSIZE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
}
