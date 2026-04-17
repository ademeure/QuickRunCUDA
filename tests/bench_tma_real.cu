// HONEST TMA bench: each TMA has UNIQUE smem region AND unique source offset,
// so we're actually testing real L2/DRAM bandwidth.
// Max smem per CTA (no opt-in) on B300 = ~200 KB. Bounded by SMEM_BUDGET_KB.
//
// Uses single mbarrier, fires NT TMAs, waits, repeats.
// Forces ld.shared.v4 of all NT regions after barrier (strong data dep).

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef NTMAS
#define NTMAS 16
#endif
#ifndef SRC_STRIDE
#define SRC_STRIDE 65536
#endif
// SMEM_STRIDE = TMA_BYTES (each TMA gets its own smem region)

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
    unsigned int data_xor = 0;
    unsigned long long total_polls = 0;

    if (threadIdx.x == 0) {
        unsigned phase = 0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            // Source stride per-iter to avoid L1-trivial reuse
            unsigned long long iter_off = ((unsigned long long)i * SRC_STRIDE) & 0x3FFFFFFFull;
            unsigned long long src_base = (unsigned long long)A + iter_off;

            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr),
                   "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr + k * TMA_BYTES),   // UNIQUE smem offset
                       "l"(src_base + k * TMA_BYTES),    // UNIQUE source offset
                       "r"(mbar_addr),
                       "n"((unsigned)TMA_BYTES) : "memory");
            }

            // Busy-poll test_wait
            unsigned p = 0;
            #pragma unroll 1
            for (int t = 0; t < 100000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
                total_polls++;
            }
            phase ^= 1;

            // Force dep on ALL TMAs' data (checks they actually landed distinctly)
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(x) : "r"(smem_addr + k * TMA_BYTES));
                data_xor ^= x;
            }
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned long long*)C)[1] = total_polls;
        ((unsigned int*)C)[4] = data_xor;
    }
}
