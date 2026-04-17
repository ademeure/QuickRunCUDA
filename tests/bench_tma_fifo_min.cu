// Minimal 2-barrier FIFO — no arrays, hardcoded slots.
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bar0;
    __shared__ __align__(8) unsigned long long bar1;
    unsigned bar0_addr = (unsigned)__cvta_generic_to_shared(&bar0);
    unsigned bar1_addr = (unsigned)__cvta_generic_to_shared(&bar1);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar0_addr));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar1_addr));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

    if (threadIdx.x == 0) {
        unsigned phase0 = 0, phase1 = 0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        // Prime: arrive+issue to bar0 (first in flight)
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
            :: "r"(bar0_addr), "n"((unsigned)TMA_BYTES) : "memory");
        asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];"
            :: "r"(smem_addr), "l"((unsigned long long)A),
               "r"(bar0_addr), "n"((unsigned)TMA_BYTES) : "memory");

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            // ping-pong: issue on opposite bar, wait on other
            if ((i & 1) == 0) {
                // Issue to bar1 while waiting on bar0
                unsigned long long src = (unsigned long long)A + ((unsigned long long)(i+1) * TMA_BYTES & 0x3FFFFFFFull);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                    :: "r"(bar1_addr), "n"((unsigned)TMA_BYTES) : "memory");
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr + TMA_BYTES), "l"(src),
                       "r"(bar1_addr), "n"((unsigned)TMA_BYTES) : "memory");
                // Wait on bar0
                unsigned p = 0;
                for (int t = 0; t < 10000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar0_addr), "r"(phase0));
                }
                phase0 ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                unsigned x;
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr));
                data_xor ^= x;
            } else {
                unsigned long long src = (unsigned long long)A + ((unsigned long long)(i+1) * TMA_BYTES & 0x3FFFFFFFull);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                    :: "r"(bar0_addr), "n"((unsigned)TMA_BYTES) : "memory");
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr), "l"(src),
                       "r"(bar0_addr), "n"((unsigned)TMA_BYTES) : "memory");
                unsigned p = 0;
                for (int t = 0; t < 10000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar1_addr), "r"(phase1));
                }
                phase1 ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                unsigned x;
                asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + TMA_BYTES));
                data_xor ^= x;
            }
        }

        // Drain last in-flight (bar0 or bar1, whichever is pending)
        if ((ITERS & 1) == 0) {
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar0_addr), "r"(phase0));
            }
        } else {
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar1_addr), "r"(phase1));
            }
        }

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[4] = data_xor;
    }
}
