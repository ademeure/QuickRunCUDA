// V6 I4: Multicast TMA (cp.async.bulk.shared::cluster.multicast)
// Copy once from global to multiple CTAs' shared memory in a cluster
// MODE 0: regular cp.async.bulk.shared::cluster (no multicast)
// MODE 1: multicast to all CTAs in cluster (mask = (1 << CSIZE) - 1)
#ifndef MODE
#define MODE 0
#endif

#define CSIZE 4

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(CSIZE, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(16) unsigned int smem[256];
    __shared__ __align__(8) unsigned long long bar;
    unsigned int bar_addr = __cvta_generic_to_shared(&bar);
    unsigned int smem_addr = __cvta_generic_to_shared(smem);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    unsigned long long t0, t1;
    if (threadIdx.x == 0 && blockIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned int bytes = 1024;  // 1 KB copy
    unsigned int cmask = (1 << CSIZE) - 1;  // all CTAs in cluster

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Regular cp.async.bulk to local CTA's smem
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                         :: "r"(bar_addr), "r"(bytes));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1], %2, [%3];"
                         :: "r"(smem_addr), "l"(A + i * 32), "r"(bytes), "r"(bar_addr));
        }
#elif MODE == 1
        // Multicast to all CTAs in cluster (only block 0 issues)
        if (threadIdx.x == 0 && blockIdx.x == 0) {
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                         :: "r"(bar_addr), "r"(bytes));
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes.multicast::cluster "
                         "[%0], [%1], %2, [%3], %4;"
                         :: "r"(smem_addr), "l"(A + i * 32), "r"(bytes), "r"(bar_addr), "h"((unsigned short)cmask));
        }
#endif
    }

    if (threadIdx.x == 0 && blockIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[0] == 0xDEADBEEF) C[blockIdx.x] = (float)smem[0];
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d ITERS=%d cy/iter=%.3f\n", MODE, ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
