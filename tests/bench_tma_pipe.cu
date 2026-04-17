// FIFO pipeline of N mbarriers vs single mbarrier at N NTMAS.
// Compare time/BW for SAME total in-flight bytes.
//
// OP 0 = single mbarrier, NTMAS TMAs, wait per iter
// OP 1 = FIFO: DEPTH mbarriers, 1 TMA each, arrive oldest, wait oldest→age

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef NTMAS
#define NTMAS 8
#endif
#ifndef DEPTH
#define DEPTH 8
#endif
#ifndef OP
#define OP 0
#endif
#ifndef SRC_STRIDE
#define SRC_STRIDE 4096
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bars[64];
    unsigned bar_base = (unsigned)__cvta_generic_to_shared(&bars[0]);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    int n_bars = (OP == 0) ? 1 : DEPTH;
    if (threadIdx.x == 0) {
        for (int b = 0; b < n_bars; b++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_base + b*8));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

    if (threadIdx.x == 0) {
        unsigned phase_s = 0;
        unsigned phases[64] = {0};
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if OP == 0
        // Single mbarrier, NT TMAs per iter, wait per iter
        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            unsigned long long src_base = (unsigned long long)A
                + (((unsigned long long)i * NTMAS * SRC_STRIDE) & 0x3FFFFFFFull);

            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base), "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr + k * TMA_BYTES),
                       "l"(src_base + k * SRC_STRIDE),
                       "r"(bar_base),
                       "n"((unsigned)TMA_BYTES) : "memory");
            }
            unsigned p = 0;
            #pragma unroll 1
            for (int t = 0; t < 100000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base), "r"(phase_s));
            }
            phase_s ^= 1;

            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(x) : "r"(smem_addr + k * TMA_BYTES));
                data_xor ^= x;
            }
        }
#else
        // FIFO pipeline: DEPTH mbarriers, 1 TMA each.
        // Issue TMA on bar[i%DEPTH], then wait on bar[(i - DEPTH + 1) % DEPTH] (oldest in flight)
        #pragma unroll 1
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned long long src = (unsigned long long)A
                + (((unsigned long long)i * SRC_STRIDE) & 0x3FFFFFFFull);

            asm volatile(
                "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + slot * TMA_BYTES),
                   "l"(src),
                   "r"(bar_base + slot*8),
                   "n"((unsigned)TMA_BYTES) : "memory");

            if (i >= DEPTH - 1) {
                int old = (i - DEPTH + 1) % DEPTH;
                unsigned p = 0;
                #pragma unroll 1
                for (int t = 0; t < 100000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar_base + old*8), "r"(phases[old]));
                }
                phases[old] ^= 1;
                asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(x) : "r"(smem_addr + old * TMA_BYTES));
                data_xor ^= x;
            }
        }
        // drain
        for (int s = 0; s < DEPTH; s++) {
            unsigned p = 0;
            for (int t = 0; t < 100000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base + s*8), "r"(phases[s]));
            }
        }
#endif

        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[4] = data_xor;
    }
}
