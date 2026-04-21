// V7 I3: cp.async vs cp.async.bulk for different transfer sizes
// MODE 0: cp.async per thread (16 B fixed; 32 threads × 16 = 512 B per warp per iter)
// MODE 1: cp.async.bulk (BYTES set per call; single-thread issue, 32-warp delivery via mbarrier)
#ifndef MODE
#define MODE 0
#endif
#ifndef BYTES
#define BYTES 1024
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    __shared__ __align__(16) unsigned int smem[4096];  // 16 KB
    __shared__ __align__(8) unsigned long long bar;
    unsigned int smem_addr_base = (unsigned int)__cvta_generic_to_shared(smem);
    unsigned int bar_addr = __cvta_generic_to_shared(&bar);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_addr));
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncwarp();

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // cp.async per thread, 16 B each (max for cp.async)
        // BYTES total = 32 × 16 = 512 B per iteration regardless
        // For BYTES > 512: scale with multiple cp.async per thread
        int n_per_thr = BYTES / 32 / 16;  // bytes per thread / 16 B per cp.async
        if (n_per_thr < 1) n_per_thr = 1;
        #pragma unroll 1
        for (int k = 0; k < n_per_thr; k++) {
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
                         "cp.async.commit_group;"
                         :: "r"(smem_addr_base + (threadIdx.x + k * 32) * 16),
                            "l"(A + ((i * (n_per_thr*32)) + threadIdx.x + k * 32) * 4));
        }
        asm volatile("cp.async.wait_all;");
#elif MODE == 1
        // cp.async.bulk single-thread issue
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                         :: "r"(bar_addr), "r"((unsigned)BYTES));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                         "[%0], [%1], %2, [%3];"
                         :: "r"(smem_addr_base), "l"(A + i * BYTES / 4), "r"((unsigned)BYTES), "r"(bar_addr));
        }
        asm volatile("{ .reg .pred p;\n"
                     "L_w_%=: mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1;\n"
                     "  @!p bra L_w_%=; }"
                     :: "r"(bar_addr), "r"(i & 1));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (smem[threadIdx.x] == 0xCAFEBABE) C[blockIdx.x] = (float)smem[threadIdx.x];

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("MODE=%d BYTES=%d ITERS=%d cy/iter=%.3f cy/byte=%.4f\n",
               MODE, BYTES, ITERS, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS/(double)BYTES);
    }
}
