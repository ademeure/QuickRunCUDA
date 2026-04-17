// GEMM-style TMA: warp 0 issues, ALL threads in block read the tile.
// Each slot has its own mbarrier.
//
// Each consumer "reads" its stripe and xor-accumulates (128-bit per lane).
// This is the canonical load-tile-then-compute-tile pattern.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 16384
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
            // empty[] init count=BLOCK_SIZE so each consumer thread's arrive counts
            asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;"
                :: "r"(empty_base + b*8), "r"((unsigned)BLOCK_SIZE));
        }
    }
    __syncthreads();
    // Each thread pre-arrives empty[] (count=BLOCK_SIZE × DEPTH primes, so all are "empty" initially)
    for (int b = 0; b < DEPTH; b++) {
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(empty_base + b*8));
    }
    asm volatile("fence.proxy.async.shared::cta;");

    unsigned int data_xor = 0;
    unsigned long long t0 = 0, t1 = 0;

    // Single timer on thread 0
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // All threads run the loop
    unsigned c_ph = 0;  // per-thread tracking: phase bit for full[]
    unsigned p_ph = 0;  // phase bit for empty[]

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        int slot = i % DEPTH;

        // Producer (thread 0): wait for empty[slot] to be released by all threads, then issue TMA
        if (threadIdx.x == 0) {
            unsigned target = (p_ph >> slot) & 1;
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(empty_base + slot*8), "r"(target));
            }
            p_ph ^= (1u << slot);
            unsigned long long src = (unsigned long long)A + (((unsigned long long)i * TMA_BYTES) & 0x3FFFFFFFull);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(full_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + slot * TMA_BYTES), "l"(src),
                   "r"(full_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
        }

        // All consumers wait on full[slot]
        unsigned target_f = (c_ph >> slot) & 1;
        unsigned p = 0;
        for (int t = 0; t < 10000 && !p; t++) {
            asm volatile(
                "{ .reg .pred P; mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; selp.b32 %0, 1, 0, P; }"
                : "=r"(p) : "r"(full_base + slot*8), "r"(target_f));
        }
        c_ph ^= (1u << slot);

        // Each thread reads its stripe of the tile (TMA_BYTES/BLOCK_SIZE bytes per thread)
        // For TMA_BYTES=16KB, BLOCK_SIZE=128 → 128 bytes/thread = 32 u32 loads
        const int per_thread_u32 = (TMA_BYTES / BLOCK_SIZE) / 4;
        unsigned int acc = 0;
        #pragma unroll
        for (int k = 0; k < per_thread_u32; k++) {
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];"
                : "=r"(x) : "r"(smem_addr + slot * TMA_BYTES + (threadIdx.x * per_thread_u32 + k) * 4));
            acc ^= x;
        }
        data_xor ^= acc;

        // All arrive on empty[slot]
        asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(empty_base + slot*8));
    }

    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x * 3 + 0] = t1 - t0;
    }
    // write xor from arbitrary thread to prevent DCE
    if ((int)data_xor == seed)
        ((unsigned int*)C)[blockIdx.x * 1024 + threadIdx.x] = data_xor;
}
