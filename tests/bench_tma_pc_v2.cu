// Same prod/cons as bench_tma_pc.cu but consumer uses FULL WARP 1 for reads.
// Tests whether 4 KB consumer-bound plateau lifts when ld.shared is parallelized.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef DEPTH
#define DEPTH 4
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long full[DEPTH];
    __shared__ __align__(8) unsigned long long empty[DEPTH];
    unsigned full_base  = (unsigned)__cvta_generic_to_shared(&full[0]);
    unsigned empty_base = (unsigned)__cvta_generic_to_shared(&empty[0]);
    unsigned smem_addr  = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        #pragma unroll
        for (int b = 0; b < DEPTH; b++) {
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(full_base  + b*8));
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(empty_base + b*8));
        }
        #pragma unroll
        for (int b = 0; b < DEPTH; b++)
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(empty_base + b*8));
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        // PRODUCER
        unsigned p_ph = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned target = (p_ph >> slot) & 1;
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(empty_base + slot*8), "r"(target));
            }
            p_ph ^= (1u << slot);

            unsigned long long src = (unsigned long long)A
                + (((unsigned long long)i * TMA_BYTES) & 0x3FFFFFFFull);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(full_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + slot * TMA_BYTES), "l"(src),
                   "r"(full_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
        }
    } else if (wid == 1) {
        // FULL-WARP CONSUMER (all 32 lanes)
        unsigned c_ph = 0;
        unsigned int local_xor = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            // Only lane 0 does test_wait
            unsigned p = 0;
            if (lid == 0) {
                unsigned target = (c_ph >> slot) & 1;
                for (int t = 0; t < 10000 && !p; t++) {
                    asm volatile(
                        "{ .reg .pred P; "
                        "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                        "selp.b32 %0, 1, 0, P; }"
                        : "=r"(p) : "r"(full_base + slot*8), "r"(target));
                }
            }
            // Broadcast wait-done via __syncwarp (cheap, ~8 cy)
            __syncwarp();
            if (lid == 0) c_ph ^= (1u << slot);

            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            // Full warp reads data (each lane 128 bytes of the 4KB/32KB tile)
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];"
                : "=r"(x) : "r"(smem_addr + slot * TMA_BYTES + lid * 4));
            local_xor ^= x;

            // Only lane 0 signals empty
            __syncwarp();
            if (lid == 0)
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(empty_base + slot*8));
        }
        // Reduce local_xor across warp (not needed for BW; just prevents DCE)
        for (int off = 16; off > 0; off >>= 1)
            local_xor ^= __shfl_xor_sync(0xFFFFFFFFu, local_xor, off);
        if (lid == 0) data_xor = local_xor;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x * 3 + 0] = t1 - t0;
        ((unsigned int*)C)[blockIdx.x * 6 + 5] = data_xor;
    }
}
