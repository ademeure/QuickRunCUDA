// 4 KiB TMA batched on ONE mbarrier — amortize consumer overhead across N TMAs.
// Producer: per iter, arrive+expect_tx = N * TMA_BYTES, fire N cp.async.bulk
// Consumer: wait once, then reads ALL N smem regions
// Each TMA gets unique smem region (N * TMA_BYTES smem total)

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef NTMAS
#define NTMAS 16
#endif
#ifndef DEPTH
#define DEPTH 1
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long full[DEPTH];
    __shared__ __align__(8) unsigned long long empty[DEPTH];
    unsigned fb = (unsigned)__cvta_generic_to_shared(&full[0]);
    unsigned eb = (unsigned)__cvta_generic_to_shared(&empty[0]);
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        for (int b = 0; b < DEPTH; b++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(fb+b*8));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(eb+b*8));
            asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(eb+b*8));
        }
    }
    __syncthreads();

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        unsigned p_ph = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned target = (p_ph >> slot) & 1;
            unsigned p = 0;
            while (!p) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; selp.b32 %0, 1, 0, P; }" : "=r"(p) : "r"(eb+slot*8), "r"(target));
            p_ph ^= (1u << slot);
            asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(fb+slot*8));
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                :: "r"(fb+slot*8), "n"((unsigned)(NTMAS * TMA_BYTES)));
            // Per-CTA unique start + per-iter consecutive NTMAS*TMA_BYTES stride.
            // Same methodology as bench_tma_acquire_v2 for apples-to-apples L2 residency.
            unsigned long long off = ((unsigned long long)blockIdx.x * NTMAS * TMA_BYTES * DEPTH
                                    + (unsigned long long)i * NTMAS * TMA_BYTES) & 0x3FE00000ull;
            unsigned long long base_src = (unsigned long long)A + off;
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];"
                    :: "r"(sa + slot * NTMAS * TMA_BYTES + k * TMA_BYTES),
                       "l"(base_src + k * TMA_BYTES),
                       "r"(fb+slot*8),
                       "n"((unsigned)TMA_BYTES) : "memory");
            }
        }
    } else if (wid == 1 && lid == 0) {
        unsigned c_ph = 0;
        unsigned int local_xor = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned target = (c_ph >> slot) & 1;
            unsigned p = 0;
            while (!p) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; selp.b32 %0, 1, 0, P; }" : "=r"(p) : "r"(fb+slot*8), "r"(target));
            c_ph ^= (1u << slot);
            // Read first u32 of each TMA region (forces data dep)
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(x) : "r"(sa + slot * NTMAS * TMA_BYTES + k * TMA_BYTES));
                local_xor ^= x;
            }
            asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(eb+slot*8));
        }
        data_xor = local_xor;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
    }
    if ((int)data_xor == seed) ((unsigned int*)C)[1] = data_xor;
}
