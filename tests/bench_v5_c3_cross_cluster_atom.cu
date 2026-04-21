// V5 C3: Cross-cluster atomics — atomic on DSMEM
// Compare:
//   MODE 0: atom.shared::cta (local SMEM atomic) — baseline
//   MODE 1: atom.shared::cluster (peer CTA's SMEM via mapa)
//   MODE 2: atom.global (HBM atomic) — for ratio
// All threads in cluster bombard a single counter
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(128, 1)
__cluster_dims__(2, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned int counter;
    unsigned int local_addr = __cvta_generic_to_shared(&counter);

    if (blockIdx.x == 0 && threadIdx.x == 0) counter = 0;
    asm volatile("barrier.cluster.arrive.aligned;");
    asm volatile("barrier.cluster.wait.aligned;");

    // Resolve target address
    unsigned int target_addr_smem;
#if MODE == 0 || MODE == 1
#if MODE == 1
    asm volatile("mapa.shared::cluster.u32 %0, %1, 0;"
                 : "=r"(target_addr_smem) : "r"(local_addr));
#else
    target_addr_smem = local_addr;
#endif
#endif

    unsigned long long t0, t1;
    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    unsigned int my_val = threadIdx.x + 1;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // CTA-local SMEM atomic — only CTA 0 hits its local; CTA 1 hits its own (no contention across CTAs)
        unsigned int old;
        asm volatile("atom.shared::cta.add.u32 %0, [%1], %2;"
                     : "=r"(old) : "r"(target_addr_smem), "r"(my_val));
#elif MODE == 1
        // Cross-cluster atomic: CTA 0 hits local, CTA 1 hits peer (CTA 0's bar)
        unsigned int old;
        asm volatile("atom.shared::cluster.add.u32 %0, [%1], %2;"
                     : "=r"(old) : "r"(target_addr_smem), "r"(my_val));
#elif MODE == 2
        // Global memory atomic — all 256 threads contend on A[0]
        unsigned int old = atomicAdd((unsigned int*)A, my_val);
#endif
        // Anti-DCE
        if (i == ITERS - 1 && threadIdx.x == 0) C[blockIdx.x] = (float)i;
    }

    if (blockIdx.x == 0 && threadIdx.x == 0)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned int total = (MODE == 2) ? *(unsigned int*)A : counter;
        printf("MODE=%d clk=%llu cy/iter=%.3f total=%u\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS, total);
    }
}
