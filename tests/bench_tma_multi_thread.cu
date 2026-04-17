// Multi-thread TMA issue: each of N warp-leaders fires its own TMA.
// Compare vs single-thread issuing N TMAs serially.
//
// OP 0 = single-thread issues NTMAS TMAs (baseline)
// OP 1 = NTMAS warp-leaders (tid%32==0) each issue 1 TMA to shared mbarrier
// OP 2 = NTMAS warp-leaders, each to its own mbarrier

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef NTMAS
#define NTMAS 8
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bars[16];
    unsigned bar_base = (unsigned)__cvta_generic_to_shared(&bars[0]);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < 16; b++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_base + b*8));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

    int lid = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned long long src_base = (unsigned long long)A +
            (((unsigned long long)i * NTMAS * TMA_BYTES) & 0x3FFFFFFFull);

#if OP == 0  // single-thread issues NTMAS TMAs
        if (threadIdx.x == 0) {
            static unsigned phase = 0;
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base), "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                    " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr + k * TMA_BYTES),
                       "l"(src_base + k * TMA_BYTES),
                       "r"(bar_base), "n"((unsigned)TMA_BYTES) : "memory");
            }
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base), "r"(phase));
            }
            phase ^= 1;
        }
        __syncthreads();
        // Force data dep via first thread
        if (threadIdx.x == 0) {
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(x) : "r"(smem_addr + k * TMA_BYTES));
                data_xor ^= x;
            }
        }

#elif OP == 1  // NTMAS warp leaders issue to SHARED mbarrier (count=NTMAS)
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                :: "r"(bar_base), "r"((unsigned)NTMAS));
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base), "n"((unsigned)(NTMAS * TMA_BYTES)) : "memory");
        }
        __syncthreads();
        if (lid == 0 && warp_id < NTMAS) {
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + warp_id * TMA_BYTES),
                   "l"(src_base + warp_id * TMA_BYTES),
                   "r"(bar_base), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(bar_base));
        }
        if (threadIdx.x == 0) {
            static unsigned phase = 0;
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base), "r"(phase));
            }
            phase ^= 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(x) : "r"(smem_addr + k * TMA_BYTES));
                data_xor ^= x;
            }
        }
        __syncthreads();

#elif OP == 2  // NTMAS warp leaders, EACH on own mbarrier
        if (lid == 0 && warp_id < NTMAS) {
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base + warp_id*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile(
                "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + warp_id * TMA_BYTES),
                   "l"(src_base + warp_id * TMA_BYTES),
                   "r"(bar_base + warp_id*8),
                   "n"((unsigned)TMA_BYTES) : "memory");
        }
        __syncthreads();
        // leader waits on all
        if (threadIdx.x == 0) {
            static unsigned phase2 = 0;
            for (int s = 0; s < NTMAS; s++) {
                unsigned p = 0;
                for (int t = 0; t < 10000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(bar_base + s*8), "r"(phase2));
                }
            }
            phase2 ^= 1;
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(x) : "r"(smem_addr + k * TMA_BYTES));
                data_xor ^= x;
            }
        }
        __syncthreads();
#endif
    }

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[4] = data_xor;
    }
}
