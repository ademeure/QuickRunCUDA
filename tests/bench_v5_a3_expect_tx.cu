// V5 A3: mbarrier.expect_tx + cp.async.bulk pattern
// Modern Hopper+ async copy with byte-counted transaction barrier
// Compare 3 patterns:
//   MODE 0: cp.async.cg + commit_group + wait_all (legacy)
//   MODE 1: cp.async.bulk.shared.global + mbarrier.expect_tx + arrive + wait
//   MODE 2: TMA via cp.async.bulk.tensor (advanced)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[1024];
    __shared__ __align__(8) unsigned long long bar;
    unsigned int bar_addr = __cvta_generic_to_shared(&bar);
    unsigned int smem_addr = __cvta_generic_to_shared(smem);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncwarp();

    unsigned long long t0, t1;
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Legacy cp.async + commit + wait
        asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                     "cp.async.commit_group;\n"
                     "cp.async.wait_all;"
                :: "r"(smem_addr + threadIdx.x * 16),
                   "l"(A + (i * 32 + threadIdx.x) * 4));
#elif MODE == 1
        // cp.async.bulk + transaction barrier (Hopper+ pattern)
        // Single thread issues bulk; expect_tx tracks bytes; arrive on completion
        if (threadIdx.x == 0) {
            unsigned int bytes = 32 * 16;  // 32 threads * 16 B each
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                         :: "r"(bar_addr), "r"(bytes));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1], %2, [%3];"
                :: "r"(smem_addr),
                   "l"(A + i * 32 * 4),
                   "r"(bytes),
                   "r"(bar_addr));
        }
        // All threads wait
        asm volatile("{ .reg .pred p;\n"
                     "L_w_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], 0;\n"
                     "  @!p bra L_w_%=; }"
                :: "r"(bar_addr));
#endif
    }

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[0] == 0xDEADBEEF) C[blockIdx.x] = (float)smem[0];
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
