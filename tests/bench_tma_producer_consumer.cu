// Producer/consumer pattern: warp 0 issues TMAs, warps 1..K wait+read.
// This should reveal true engine pipeline depth (issue isn't serialized with waits).
//
// OP 0 = producer thread 0 only; consumers = warp-leaders of warps 1..
// OP 1 = SINGLE-thread classic (for comparison)

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 256
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 4096
#endif
#ifndef DEPTH
#define DEPTH 8
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ __align__(128) unsigned char smem_buf[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ __align__(8) unsigned long long bars[DEPTH];
    __shared__ volatile int produce_cnt;  // head: next slot to issue
    __shared__ volatile int consume_cnt;  // tail: next slot to retire
    unsigned bar_base = (unsigned)__cvta_generic_to_shared(&bars[0]);
    unsigned smem_addr = (unsigned)__cvta_generic_to_shared(smem_buf);

    if (threadIdx.x == 0) {
        for (int b = 0; b < DEPTH; b++)
            asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(bar_base + b*8));
        produce_cnt = 0;
        consume_cnt = 0;
    }
    __syncthreads();
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");

    unsigned long long t0 = 0, t1 = 0;
    unsigned int data_xor = 0;

#if OP == 1
    // Classic single-thread — thread 0 does everything, ITERS-wide
    if (threadIdx.x == 0) {
        unsigned phase_bits = 0;
        // Prime DEPTH-1
        for (int i = 0; i < DEPTH - 1; i++) {
            unsigned long long src = (unsigned long long)A + ((unsigned long long)i * TMA_BYTES & 0x3FFFFFFFull);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base + i*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + i * TMA_BYTES), "l"(src),
                   "r"(bar_base + i*8), "n"((unsigned)TMA_BYTES) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        for (int i = DEPTH - 1; i < ITERS + DEPTH - 1; i++) {
            int slot = i % DEPTH;
            int old = (i + 1) % DEPTH;
            if (i < ITERS) {
                unsigned long long src = (unsigned long long)A + ((unsigned long long)i * TMA_BYTES & 0x3FFFFFFFull);
                asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                    :: "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
                asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                             " [%0], [%1], %3, [%2];"
                    :: "r"(smem_addr + slot * TMA_BYTES), "l"(src),
                       "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            }
            unsigned target = (phase_bits >> old) & 1;
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base + old*8), "r"(target));
            }
            phase_bits ^= (1u << old);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + old * TMA_BYTES));
            data_xor ^= x;
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    }
    __syncthreads();
    if (threadIdx.x == 0) ((unsigned long long*)C)[0] = t1 - t0;

#else
    // Producer = thread 0. Consumer = tid/32 warp, lane 0.
    // Producer issues TMA to slot i%DEPTH, ONLY if (head - tail < DEPTH), else wait for tail to advance.
    // Consumer waits on oldest outstanding (tail slot), increments consume_cnt.
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    if (threadIdx.x == 0) {
        // Producer
        unsigned prod_phase = 0;  // tracks arm phase per slot (for reuse)
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            // Wait for slot to be free (consumed)
            while (produce_cnt - consume_cnt >= DEPTH) { /* spin */ }
            unsigned long long src = (unsigned long long)A + ((unsigned long long)i * TMA_BYTES & 0x3FFFFFFFull);
            asm volatile("mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 _, [%0], %1;"
                :: "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes"
                         " [%0], [%1], %3, [%2];"
                :: "r"(smem_addr + slot * TMA_BYTES), "l"(src),
                   "r"(bar_base + slot*8), "n"((unsigned)TMA_BYTES) : "memory");
            asm volatile("" ::: "memory");
            produce_cnt = i + 1;
        }
    } else if (wid == 1 && lid == 0) {
        // Consumer warp-leader
        unsigned cons_phase_bits = 0;
        unsigned int local_xor = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            // Wait for producer to have issued this slot
            while (produce_cnt <= i) { /* spin */ }
            unsigned target = (cons_phase_bits >> slot) & 1;
            unsigned p = 0;
            for (int t = 0; t < 10000 && !p; t++) {
                asm volatile(
                    "{ .reg .pred P; "
                    "mbarrier.test_wait.parity.shared::cta.b64 P, [%1], %2; "
                    "selp.b32 %0, 1, 0, P; }"
                    : "=r"(p) : "r"(bar_base + slot*8), "r"(target));
            }
            cons_phase_bits ^= (1u << slot);
            asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
            unsigned int x;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(x) : "r"(smem_addr + slot * TMA_BYTES));
            local_xor ^= x;
            asm volatile("" ::: "memory");
            consume_cnt = i + 1;
        }
        data_xor = local_xor;
    }
    __syncthreads();
    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[4] = data_xor;
    }
#endif
}
