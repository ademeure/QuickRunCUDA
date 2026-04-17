// TMA throughput with *forced* DRAM reads.
// Working set = ITERS * NTMAS * TMA_BYTES per CTA. If this exceeds ~L2/CTA (~860 KB
// with 148 CTAs sharing 126 MB L2) we'll hit DRAM.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 65536
#endif
#ifndef NTMAS
#define NTMAS 4
#endif
#ifndef STRIDE
#define STRIDE 4194304  // 4 MB — guarantees each iter hits a fresh line
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long mbar;
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);

    if (threadIdx.x == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;

    if (threadIdx.x == 0) {
        unsigned phase = 0;
        unsigned long long mask = 0x3FFFFFFFull; // 1 GB
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            // Each CTA takes a unique per-iter STRIDE stride; DRAM-miss guaranteed.
            unsigned long long off = ((unsigned long long)blockIdx.x * NTMAS * TMA_BYTES
                                    + (unsigned long long)i * STRIDE) & mask;
            unsigned long long src_base = (unsigned long long)A + off;

            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr + k * TMA_BYTES),
                       "l"(src_base + k * TMA_BYTES),
                       "r"(mbar_addr),
                       "n"((unsigned)TMA_BYTES) : "memory");
            }

            unsigned p = 0;
            for (int t = 0; t < 100000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
            }
            phase ^= 1;
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0)
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
}
