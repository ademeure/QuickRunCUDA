// TMA chip-wide throughput: sweep (CTAs, NTMAS per iter, TMA_BYTES).
// Each CTA issues NTMAS bulk loads, waits, and repeats ITERS times.
// Uses one mbarrier per CTA; expect_tx set once per iter to NTMAS*TMA_BYTES.
// Total bytes per CTA per iter = NTMAS * TMA_BYTES  (≤ ~1 MB = expect_tx max).

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef NTMAS
#define NTMAS 8
#endif
#ifndef SMEM_STRIDE
#define SMEM_STRIDE TMA_BYTES
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long mbar;

    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;

    if (threadIdx.x == 0) {
        unsigned phase = 0;
        // CTA walks own swath of A; slight per-CTA offset so they don't all hit same line.
        unsigned long long cta_off = ((unsigned long long)blockIdx.x * NTMAS * TMA_BYTES) & 0x3FFFFFFFull;

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            unsigned long long iter_off = (cta_off +
                ((unsigned long long)i * gridDim.x * NTMAS * TMA_BYTES)) & 0x3FFFFFFFull;
            unsigned long long src_base = (unsigned long long)A + iter_off;

            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");

            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr + k * SMEM_STRIDE),
                       "l"(src_base + k * TMA_BYTES),
                       "r"(mbar_addr),
                       "n"((unsigned)TMA_BYTES) : "memory");
            }

            // bounded wait (safety)
            unsigned p = 0;
            for (int tries = 0; tries < 100000 && !p; tries++) {
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
