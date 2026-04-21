// V5 A5: mbarrier in DSMEM — cluster-wide barrier semantics
// Test: CTA 0 hosts a mbarrier; CTAs 1..N-1 arrive on it via mapa.shared::cluster
// Compare:
//   MODE 0: local mbarrier (single CTA, 128 threads) — baseline
//   MODE 1: cluster-shared mbarrier (CTA 0 hosts; both CTAs arrive = 256 threads)
//   MODE 2: cluster.barrier (intrinsic) for comparison
#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
#define ARRIVE_COUNT 128
#define CDIMS 1
#else
#define ARRIVE_COUNT 256
#define CDIMS 2
#endif

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(CDIMS, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bar;
    unsigned int local_bar_addr = __cvta_generic_to_shared(&bar);

    // Init: only CTA 0 thread 0 inits the barrier
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "r"(local_bar_addr), "r"((unsigned int)ARRIVE_COUNT));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
#if MODE != 0
    // Cluster-wide ordering so CTA 1 sees the init before arriving
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");
#else
    __syncthreads();
#endif

    // Resolve CTA 0's bar pointer (peer addressing)
    unsigned int peer_bar_addr;
#if MODE == 1
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                 : "=r"(peer_bar_addr) : "r"(local_bar_addr));
#else
    peer_bar_addr = local_bar_addr;
#endif

    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Local mbarrier — CTA 0 only
        asm volatile("{ .reg .b64 state;\n"
                     "  mbarrier.arrive.shared::cta.b64 state, [%0]; }"
                     :: "r"(local_bar_addr));
        asm volatile("{ .reg .pred p;\n"
                     "L_w_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                     "  @!p bra L_w_%=; }"
                     :: "r"(local_bar_addr), "r"(i & 1));
#elif MODE == 1
        // Cluster-shared mbarrier: arrive remote (sink destination)
        // Wait must use LOCAL try_wait.parity on CTA 0; CTA 1 uses cluster.barrier to sync with CTA 0
        asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                     :: "r"(peer_bar_addr));
        if (blockIdx.x == 0) {
            // Only CTA 0 (host) can use try_wait.parity locally
            asm volatile("{ .reg .pred p;\n"
                         "L_w_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                         "  @!p bra L_w_%=; }"
                         :: "r"(local_bar_addr), "r"(i & 1));
        }
        // CTA 1 doesn't directly wait; use cluster.barrier to ensure ordering
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
#elif MODE == 2
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
#endif
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
