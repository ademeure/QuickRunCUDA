// FIFO pipeline with hardcoded DEPTH — used to sweep depth cleanly.
// Issues 1 TMA per iter on round-robin barriers. Waits on oldest.

#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef DEPTH
#define DEPTH 4
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bars[DEPTH];
    unsigned bar_base = (unsigned)__cvta_generic_to_shared(&bars[0]);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < DEPTH; b++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_base + b*8));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

    if (threadIdx.x == 0) {
        // Prime: arrive + issue DEPTH-1 TMAs
        unsigned phase_bits = 0;  // tracks phase per bar as bit

        // Kick off pipeline
        for (int i = 0; i < DEPTH - 1; i++) {
            int slot = i;
            unsigned long long src = (unsigned long long)A + (((unsigned long long)i * TMA_BYTES) & 0x3FFFFFFFull);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + slot * TMA_BYTES), "l"(src),
                   "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
        }

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = DEPTH - 1; i < ITERS + DEPTH - 1; i++) {
            int slot = i % DEPTH;
            int old = (i + 1) % DEPTH;  // oldest in-flight after this issue

            if (i < ITERS) {
                unsigned long long src = (unsigned long long)A + (((unsigned long long)i * TMA_BYTES) & 0x3FFFFFFFull);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                    :: "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr + slot * TMA_BYTES), "l"(src),
                       "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            }

            // Wait on oldest
            unsigned target_phase = (phase_bits >> old) & 1;
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base + old*8), "r"(target_phase));
            }
            phase_bits ^= (1u << old);

            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + old * TMA_BYTES));
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
