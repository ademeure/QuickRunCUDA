// Minimal TMA bench — isolate each source of overhead.
// OP 0: bare TMA + test_wait busy poll (busy-poll with #pragma unroll 1)
// OP 1: + fence.proxy.async
// OP 2: + ld.shared
// OP 3: + data_xor
// OP 4: TMA + try_wait.parity (may suspend; for comparison)
// OP 5: TMA only, no wait (fires into void; not a real pattern)

#ifndef TMA_BYTES
#define TMA_BYTES 128
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(32, 1)
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
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr), "l"((unsigned long long)A),
                   "r"(mbar_addr), "n"((unsigned)TMA_BYTES) : "memory");

#if OP == 5
            // no wait (but must resolve next iter's expect_tx -- will stall on second arrive)
#elif OP == 4  // try_wait.parity (suspend-capable)
            unsigned p = 0;
            #pragma unroll 1
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
                total_polls++;
            }
#else          // test_wait (non-blocking busy poll)
            unsigned p = 0;
            #pragma unroll 1
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
                total_polls++;
            }
#endif
            phase ^= 1;

#if OP >= 1 && OP != 5
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
#if OP >= 2 && OP != 5
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr));
#endif
#if OP >= 3 && OP != 5
            data_xor ^= x;
#endif
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
