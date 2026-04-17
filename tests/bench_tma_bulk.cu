// TMA bulk (non-tensor) latency / throughput / issue cost.
//
// OP 0 = LATENCY       : 1 TMA of TMA_BYTES per iter, full mbarrier RTT.
// OP 1 = THROUGHPUT    : NTMAS TMAs fired, 1 mbarrier wait per iter.
// OP 2 = ISSUE COST    : NTMAS TMAs fired, wait only at end of loop.
// OP 3 = STORE         : smem->global store (cp.async.bulk.global.shared).
// OP 4 = PREFETCH-L2   : cp.async.bulk.prefetch.L2.
//
// -s must be >= NTMAS * TMA_BYTES (dynamic smem).
// Launch 1 block / 1 thread for pure latency.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef NTMAS
#define NTMAS 1
#endif
#ifndef OP
#define OP 0
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
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            unsigned long long off = ((unsigned long long)i * 128) & 0x3FFFFFFFull;
            unsigned long long src_base = (unsigned long long)A + off;

#if OP == 0  // LATENCY: 1 TMA, full RTT
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr), "l"(src_base), "r"(mbar_addr),
                   "n"((unsigned)TMA_BYTES) : "memory");
            unsigned p;
            do {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
            } while (!p);
            phase ^= 1;

#elif OP == 1  // THROUGHPUT: NTMAS TMAs, 1 wait
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr),
                   "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");
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
            unsigned p;
            do {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
            } while (!p);
            phase ^= 1;

#elif OP == 2  // ISSUE COST: fire NTMAS, wait only after full loop
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(mbar_addr),
                   "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");
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
            // Must drain each iter to not overflow expect_tx; use try_wait.
            unsigned p;
            do {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(mbar_addr), "r"(phase));
            } while (!p);
            phase ^= 1;

#elif OP == 3  // STORE smem->global (cp.async.bulk.global.shared::cta.bulk_group)
            // store mode uses bulk_group tracking, not mbarrier
            asm volatile(
                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                :: "l"(src_base), "r"(smem_addr), "n"((unsigned)TMA_BYTES)
                : "memory");
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");

#elif OP == 4  // PREFETCH to L2
            asm volatile(
                "cp.async.bulk.prefetch.L2.global [%0], %1;"
                :: "l"(src_base), "n"((unsigned)TMA_BYTES) : "memory");
#endif
        }

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }

    __syncthreads();
    if (threadIdx.x == 0)
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
}
