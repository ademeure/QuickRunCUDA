// Extended TMA variants: STORE, PREFETCH, REDUCE, per-TMA-mbarrier.
//
// OP 0 = cp.async.bulk.global.shared::cta + commit_group/wait_group (STORE)
// OP 1 = cp.async.bulk.prefetch.L2.global (prefetch, no dest)
// OP 2 = per-TMA separate mbarrier (16 barriers, round-robin)
// OP 3 = cp.reduce.async.bulk ADD.u32 (smem→global reduce)
// OP 4 = cp.async.bulk.shared::cluster.shared::cta (cross-CTA smem inside cluster)
// OP 5 = cp.async.bulk.shared::cta.global.bulk_group (load via bulk_group)

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
#if OP == 2
    __shared__ __align__(8) unsigned long long bars[16];
    unsigned bar_base = (unsigned)__cvta_generic_to_shared(&bars[0]);
    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < 16; b++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_base + b*8));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#else
    __shared__ __align__(8) unsigned long long mbar;
    unsigned mbar_addr = (unsigned)__cvta_generic_to_shared(&mbar);
    if (threadIdx.x == 0)
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(mbar_addr));
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif

    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);
    unsigned long long t0 = 0, t1 = 0;

    if (threadIdx.x == 0) {
        unsigned phase = 0;
        unsigned phases[16] = {0};
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            unsigned long long off = ((unsigned long long)i * 128) & 0x3FFFFFFFull;
            unsigned long long gaddr = (unsigned long long)A + off;
            unsigned long long baddr = (unsigned long long)B + off;

#if OP == 0  // STORE smem→global via bulk_group
            asm volatile(
                "cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                :: "l"(baddr), "r"(smem_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");

#elif OP == 1  // PREFETCH
            asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;"
                :: "l"(gaddr), "n"((unsigned)TMA_BYTES) : "memory");

#elif OP == 2  // per-TMA separate mbarrier, 16 in flight
            int slot = i & 15;
            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + slot*TMA_BYTES), "l"(gaddr),
                   "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            // Wait for slot from 16 iters ago (if any)
            if (i >= 16) {
                int old = (i - 16) & 15;
                unsigned p = 0;
                for (int t = 0; t < 1000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar_base + old*8), "r"(phases[old]));
                }
                phases[old] ^= 1;
            }

#elif OP == 3  // REDUCE smem→global add
            asm volatile(
                "cp.reduce.async.bulk.global.shared::cta.bulk_group.add.u32"
                " [%0], [%1], %2;"
                :: "l"(baddr), "r"(smem_addr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");

#elif OP == 5  // LOAD via bulk_group (not mbarrier)
            asm volatile(
                "cp.async.bulk.shared::cta.global.bulk_group [%0], [%1], %2;"
                :: "r"(smem_addr), "l"(gaddr), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.commit_group;");
            asm volatile("cp.async.bulk.wait_group 0;");
#endif
        }

#if OP == 2
        // Drain remaining in-flight mbarriers
        for (int s = 0; s < 16 && ITERS > s; s++) {
            unsigned p = 0;
            for (int t = 0; t < 1000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base + s*8), "r"(phases[s]));
            }
        }
#endif

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0)
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
}
