// Does cp.async.bulk.prefetch.L2 warming help the subsequent TMA?
// OP 0: baseline — direct TMA from DRAM
// OP 1: prefetch + wait-a-bit + TMA (warmed L2)

#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef OP
#define OP 0
#endif
#ifndef PREFETCH_LEAD
#define PREFETCH_LEAD 8
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long mbar;
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

    if (threadIdx.x == 0) {
        unsigned phase = 0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            // DRAM-miss workload: each iter reads fresh 4 MB-stride
            unsigned long long off = ((unsigned long long)i * 4194304) & 0x3FFFFFFFull;
            unsigned long long src = (unsigned long long)A + off;

#if OP == 1
            // Prefetch LEAD iters ahead to warm L2
            if (i + PREFETCH_LEAD < ITERS) {
                unsigned long long pref_src = (unsigned long long)A
                    + (((unsigned long long)(i + PREFETCH_LEAD) * 4194304) & 0x3FFFFFFFull);
                asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                    :: "l"(pref_src), "n"((unsigned)TMA_BYTES) : "memory");
            }
#endif

            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr), "l"(src), "r"(mbar_addr),
                   "n"((unsigned)TMA_BYTES) : "memory");

            unsigned p = 0;
            for (int t = 0; t < 100000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
            }
            phase ^= 1;

            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr));
            data_xor ^= x;
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[4] = data_xor;
    }
}
