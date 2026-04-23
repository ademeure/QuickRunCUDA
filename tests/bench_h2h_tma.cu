// HEAD-TO-HEAD: max-tuned TMA cp.async.bulk read.
// Per-CTA producer/consumer with DEPTH-deep pipeline, NTMAS issues per iteration.
// Each TMA brings TMA_BYTES into smem; consumer reads first u32 of each tile to defeat DCE.
// Address pattern walks WS_BYTES to control L2 vs DRAM regime.
// Total bytes per CTA per outer iter = NTMAS * TMA_BYTES.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 16384
#endif
#ifndef NTMAS
#define NTMAS 8
#endif
#ifndef DEPTH
#define DEPTH 4
#endif
#ifndef WS_LOG2_DEFAULT
#define WS_LOG2_DEFAULT 32
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int WS_LOG2) {
    // smem layout: DEPTH × NTMAS × TMA_BYTES
    __shared__ __align__(8) unsigned long long full[DEPTH];
    __shared__ __align__(8) unsigned long long empty[DEPTH];
    unsigned fb = (unsigned)__cvta_generic_to_shared(&full[0]);
    unsigned eb = (unsigned)__cvta_generic_to_shared(&empty[0]);
    unsigned sa = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < DEPTH; b++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(fb + b*8));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(eb + b*8));
            asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(eb + b*8));
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long mask = (1ull << WS_LOG2) - 1ull;
    // bytes per CTA per outer iter
    unsigned long long bpi = (unsigned long long)NTMAS * TMA_BYTES;
    // per-CTA stride between iterations (so all CTAs cover unique tiles each iter)
    unsigned long long step = (unsigned long long)gridDim.x * bpi;

    unsigned int data_xor = 0;
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;

    if (threadIdx.x == 0) {
        // PRODUCER
        unsigned p_ph = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned target = (p_ph >> slot) & 1;
            unsigned p = 0;
            // wait until consumer marked this slot empty
            while (!p) asm volatile(
                "{ .reg .pred P; "
                "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; "
                "selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(eb + slot*8), "r"(target));
            p_ph ^= (1u << slot);

            // arrive on full, expect NTMAS*TMA_BYTES
            asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(fb + slot*8));
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;"
                :: "r"(fb + slot*8), "n"((unsigned)(NTMAS * TMA_BYTES)));

            unsigned long long off = ((unsigned long long)blockIdx.x * bpi
                                   + (unsigned long long)i * step) & mask;
            unsigned long long base = (unsigned long long)A + off;
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile(
                    "cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes "
                    "[%0], [%1], %3, [%2];"
                    :: "r"(sa + slot * NTMAS * TMA_BYTES + k * TMA_BYTES),
                       "l"(base + k * TMA_BYTES),
                       "r"(fb + slot*8),
                       "n"((unsigned)TMA_BYTES) : "memory");
            }
        }
    } else if (wid == 1) {
        // FULL-WARP CONSUMER (anti-DCE: read first u32 of each tile, parallelized over lanes)
        unsigned c_ph = 0;
        unsigned int local_xor = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned p = 0;
            if (lid == 0) {
                unsigned target = (c_ph >> slot) & 1;
                while (!p) asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(fb + slot*8), "r"(target));
            }
            __syncwarp();
            if (lid == 0) c_ph ^= (1u << slot);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

            // distribute NTMAS reads across the warp's 32 lanes
            #pragma unroll
            for (int k = 0; k < NTMAS; k += 32) {
                int kk = k + lid;
                if (kk < NTMAS) {
                    unsigned int x;
                    asm volatile("ld.shared.u32 %0, [%1];"
                        : "=r"(x)
                        : "r"(sa + slot * NTMAS * TMA_BYTES + kk * TMA_BYTES));
                    local_xor ^= x;
                }
            }
            __syncwarp();
            if (lid == 0)
                asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(eb + slot*8));
        }
        for (int off = 16; off > 0; off >>= 1)
            local_xor ^= __shfl_xor_sync(0xFFFFFFFFu, local_xor, off);
        if (lid == 0) data_xor = local_xor;
    }
    __syncthreads();
    if ((int)data_xor == seed) ((unsigned int*)C)[blockIdx.x] = data_xor;
}
