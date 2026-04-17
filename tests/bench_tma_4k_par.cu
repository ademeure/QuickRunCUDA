// 4 KiB batched with PARALLEL consumer reads (warp-cooperative dep).
// Warp 0 thread 0 = producer.
// Warp 1 lane 0 = waits on full, __syncwarp, all 32 lanes read 1 u32 each, lane 0 signals empty.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef NTMAS
#define NTMAS 24
#endif
#ifndef DEPTH
#define DEPTH 2
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
            unsigned long long base_src = (unsigned long long)A +
                (((unsigned long long)i * NTMAS * TMA_BYTES) & 0x3FFFFFFFull);
            #pragma unroll
            for (int k = 0; k < NTMAS; k++) {
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];"
                    :: "r"(sa + slot * NTMAS * TMA_BYTES + k * TMA_BYTES),
                       "l"(base_src + k * TMA_BYTES),
                       "r"(fb+slot*8),
                       "n"((unsigned)TMA_BYTES) : "memory");
            }
        }
    } else if (wid == 1) {
        // Consumer warp: lane 0 waits on full, all lanes read, lane 0 signals empty
        unsigned c_ph = 0;
        unsigned int local_xor = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            if (lid == 0) {
                unsigned target = (c_ph >> slot) & 1;
                unsigned p = 0;
                while (!p) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; selp.b32 %0, 1, 0, P; }" : "=r"(p) : "r"(fb+slot*8), "r"(target));
            }
            __syncwarp();
            if (lid == 0) c_ph ^= (1u << slot);
            // Each lane reads the first u32 of lane'th TMA region (if lane < NTMAS)
            if (lid < NTMAS) {
                unsigned int x;
                asm volatile("ld.shared.u32 %0, [%1];"
                    : "=r"(x) : "r"(sa + slot * NTMAS * TMA_BYTES + lid * TMA_BYTES));
                local_xor ^= x;
            }
            __syncwarp();
            if (lid == 0)
                asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(eb+slot*8));
        }
        // reduce local_xor across warp
        for (int off = 16; off > 0; off >>= 1)
            local_xor ^= __shfl_xor_sync(0xFFFFFFFFu, local_xor, off);
        if (lid == 0) data_xor = local_xor;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
    }
    if ((int)data_xor == seed) ((unsigned int*)C)[1] = data_xor;
}
