// Proper producer/consumer TMA pipeline with double-mbarrier ring buffer.
//
// Full-barriers: signaled by cp.async.bulk complete_tx; consumer waits.
// Empty-barriers: signaled by consumer arrive; producer waits.
//
// Producer = warp 0 thread 0.  Consumer = warp 1 thread 0.
// Timer = thread 0.  Result in C[0] = cyc.
//
// For single-thread baseline, set OP=1: thread 0 does everything.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 64   // 2 warps: producer + consumer
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef DEPTH
#define DEPTH 4
#endif
#ifndef OP
#define OP 0   // 0 = prod/cons, 1 = single-thread
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
        // Prime empty[] as "free" — arrive once so first producer-wait returns immediately.
        #pragma unroll
        for (int b = 0; b < DEPTH; b++) {
            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(empty_base + b*8));
        }
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if OP == 1
    // Single-thread classic
    if (threadIdx.x == 0) {
        unsigned p_ph = 0;
        unsigned c_ph = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned target_e = (p_ph >> slot) & 1;
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(empty_base + slot*8), "r"(target_e));
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

            unsigned target_f = (c_ph >> slot) & 1;
            p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(full_base + slot*8), "r"(target_f));
            }
            c_ph ^= (1u << slot);

            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + slot * TMA_BYTES));
            data_xor ^= x;

            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(empty_base + slot*8));
        }
    }
#else
    // PROPER prod/cons — separate warps.
    if (threadIdx.x == 0) {
        // PRODUCER
        unsigned p_ph = 0;  // bit per slot: phase expected from empty[slot]
        // After priming empty[] once, phase of each empty[slot] = 1.
        // Producer waits for phase != p_ph; starts with p_ph=0 → passes through.
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
    } else if (wid == 1 && lid == 0) {
        // CONSUMER
        unsigned c_ph = 0;
        unsigned int local_xor = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned target = (c_ph >> slot) & 1;
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(full_base + slot*8), "r"(target));
            }
            c_ph ^= (1u << slot);

            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + slot * TMA_BYTES));
            local_xor ^= x;

            asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(empty_base + slot*8));
        }
        data_xor = local_xor;
    }
    __syncthreads();
#endif

    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x * 3 + 0] = t1 - t0;
        ((unsigned int*)C)[blockIdx.x * 6 + 5] = data_xor;
    }
}
