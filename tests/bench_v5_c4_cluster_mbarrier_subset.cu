// V5 C4: cluster-shared mbarrier with SUBSET arrive count
// Test if mbarrier enables "wait for K of N CTAs" semantic via custom arrive count
//   MODE 0: mbarrier init count = CSIZE*128 (all threads in cluster)
//   MODE 1: mbarrier init count = 128 (only CTA 0 arrives)
//   MODE 2: mbarrier init count = 256 (CTA 0 + CTA 1 arrive)
//   MODE 3: mbarrier init count = (CSIZE-1)*128 (all except CTA 0 arrive)
// Measure cy/iter on CTA 0 thread 0 (which waits)
#ifndef MODE
#define MODE 0
#endif

#define CSIZE 8

#if MODE == 0
#define INIT_COUNT (CSIZE * 128)
#define N_ARRIVING_CTAS CSIZE
#elif MODE == 1
#define INIT_COUNT 128
#define N_ARRIVING_CTAS 1
#elif MODE == 2
#define INIT_COUNT 256
#define N_ARRIVING_CTAS 2
#elif MODE == 3
#define INIT_COUNT ((CSIZE - 1) * 128)
#define N_ARRIVING_CTAS (CSIZE - 1)
#endif

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(CSIZE, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bar;
    unsigned int local_bar_addr = __cvta_generic_to_shared(&bar);

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                     :: "r"(local_bar_addr), "r"((unsigned int)INIT_COUNT));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    unsigned int peer_bar_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                 : "=r"(peer_bar_addr) : "r"(local_bar_addr));

    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Arrive pattern depends on MODE
#if MODE == 0
        // All CTAs arrive
        asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                     :: "r"(peer_bar_addr));
#elif MODE == 1
        if (blockIdx.x == 0) {
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         :: "r"(peer_bar_addr));
        }
#elif MODE == 2
        if (blockIdx.x < 2) {
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         :: "r"(peer_bar_addr));
        }
#elif MODE == 3
        if (blockIdx.x > 0) {
            asm volatile("mbarrier.arrive.shared::cluster.b64 _, [%0];"
                         :: "r"(peer_bar_addr));
        }
#endif
        // CTA 0 waits locally
        if (blockIdx.x == 0) {
            asm volatile("{ .reg .pred p;\n"
                         "L_w_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                         "  @!p bra L_w_%=; }"
                         :: "r"(local_bar_addr), "r"(i & 1));
        }
        // All CTAs sync at cluster level so next iter starts cleanly
        asm volatile("barrier.cluster.arrive.aligned;");
        asm volatile("barrier.cluster.wait.aligned;");
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d init_count=%d n_arriving_CTAs=%d cy/iter=%.3f\n",
               MODE, INIT_COUNT, N_ARRIVING_CTAS, (double)(t1-t0)/(double)ITERS);
    }
}
