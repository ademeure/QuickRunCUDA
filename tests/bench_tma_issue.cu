// Pure TMA issue cost — time ONLY the issue instructions, no wait.
// Also measures mbarrier instruction costs in isolation.
//
// OP 0 = pure cp.async.bulk issue stream (wait outside timer)
// OP 1 = pure mbarrier.arrive cost (no barrier completion)
// OP 2 = pure mbarrier.arrive.expect_tx cost
// OP 3 = pure mbarrier.test_wait cost (on completed barrier)
// OP 4 = pure mbarrier.try_wait.parity cost (on completed barrier)
// OP 5 = mbarrier.init cost

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 128
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long mbar;
    __shared__ __align__(8) unsigned long long mbar2;

    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
    unsigned mbar2_addr = (unsigned)__cvta_generic_to_shared(&mbar2);

    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar2_addr));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;

    if (threadIdx.x == 0) {
#if OP == 3 || OP == 4
        // Pre-complete mbar so test/try_wait returns true immediately
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbar_addr));
#endif
#if OP == 0
        // Pre-announce total pending_tx for the whole issue burst so the
        // final drain wait is well-defined (ITERS * TMA_BYTES ≤ 1 MB).
        unsigned total_tx = (unsigned)(ITERS * TMA_BYTES);
        asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
            :: "r"(mbar_addr), "r"(total_tx) : "memory");
#endif

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i += UNROLL) {
            #pragma unroll
            for (int u = 0; u < UNROLL; u++) {
#if OP == 0  // cp.async.bulk issue only (matched by expect_tx above)
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr),
                       "l"((unsigned long long)A + ((i+u) & 0xFF) * 128),
                       "r"(mbar_addr),
                       "n"((unsigned)TMA_BYTES) : "memory");
#elif OP == 1  // mbarrier.arrive only
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(mbar_addr));
#elif OP == 2  // mbarrier.arrive.expect_tx
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], 128;"
                    :: "r"(mbar_addr));
#elif OP == 3  // test_wait (on completed barrier)
                unsigned p;
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.shared::cta.b64 P, [%1], 0; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr));
#elif OP == 4  // try_wait.parity (on completed barrier)
                unsigned p;
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], 0; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr));
#elif OP == 5  // mbarrier.init
                asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar2_addr));
#elif OP == 6  // fence.proxy.async.shared::cta
                asm volatile("fence.proxy.async.shared::cta;");
#elif OP == 7  // fence.mbarrier_init.release.cluster
                asm volatile("fence.mbarrier_init.release.cluster;");
#endif
            }
        }

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

#if OP == 0
        // Bounded drain (safety: avoid deadlock from mismatch).
        unsigned p = 0;
        unsigned long long t_timeout_start;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_timeout_start));
        for (int tries = 0; tries < 1000000 && !p; tries++) {
            asm volatile(
                "{ .reg .pred P; "
                "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], 0, 1024; "
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(mbar_addr));
            unsigned long long tn;
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(tn));
            if (tn - t_timeout_start > 50000000ULL) break;  // 25 ms cap
        }
#endif
    }

    __syncthreads();
    if (threadIdx.x == 0)
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
}
