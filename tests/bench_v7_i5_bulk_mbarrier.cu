// V7 I5: cp.async.bulk + mbarrier expect_tx with proper alternating-parity phase tracking
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[4096];
    __shared__ __align__(8) unsigned long long bar;
    unsigned int smem_addr = __cvta_generic_to_shared(smem);
    unsigned int bar_addr = __cvta_generic_to_shared(&bar);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncwarp();

    unsigned int bytes = 4096;  // 4 KB per copy

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                         :: "r"(bar_addr), "r"(bytes));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1], %2, [%3];"
                         :: "r"(smem_addr), "l"(A + i * bytes / 4), "r"(bytes), "r"(bar_addr));
        }
        // Wait — alternate parity 0,1,0,1,...
        int parity = i & 1;
        asm volatile("{ .reg .pred p;\n"
                     "L_w_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                     "  @!p bra L_w_%=; }"
                     :: "r"(bar_addr), "r"(parity));
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[0] == 0xDEADBEEF) C[blockIdx.x] = (float)smem[0];
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("ITERS=%d cy/iter=%.3f (4 KB bulk copy w/ mbarrier)\n",
               ITERS, (double)(t1-t0)/(double)ITERS);
    }
}
