// HMMA × TMA overlap test. Warp 0 issues TMA, warps 1-3 run HMMA chain.
// Compare timing of: HMMA alone, TMA alone, HMMA+TMA concurrent.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef HMMA_PER_ITER
#define HMMA_PER_ITER 8
#endif
#ifndef OP
#define OP 0
#endif
#ifndef TMA_BYTES
#define TMA_BYTES 65536
#endif
#ifndef DEPTH
#define DEPTH 3
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

    unsigned long long t0, t1;
    int wid = threadIdx.x >> 5;
    int lid = threadIdx.x & 31;

    // FP16 accumulator-relevant registers for HMMA
    unsigned acc[4] = {0};
    unsigned a_reg[4] = {threadIdx.x, threadIdx.x+1, threadIdx.x+2, threadIdx.x+3};
    unsigned b_reg[2] = {threadIdx.x*3, threadIdx.x*5};

    if (threadIdx.x == 0) asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

#if OP == 0 || OP == 2
    // Warp 0: TMA producer (OP 0 = TMA only; OP 2 = TMA + HMMA)
    if (threadIdx.x == 0 && wid == 0) {
        unsigned pph = 0;
        for (int i = 0; i < ITERS; i++) {
            int slot = i % DEPTH;
            unsigned t = (pph >> slot) & 1;
            unsigned p = 0;
            while (!p) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; selp.b32 %0, 1, 0, P; }" : "=r"(p) : "r"(eb+slot*8), "r"(t));
            pph ^= (1u << slot);
            unsigned long long src = (unsigned long long)A + (((unsigned long long)i * TMA_BYTES) & 0x3FFFFFFFull);
            asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(fb+slot*8));
            asm volatile("mbarrier.expect_tx.shared::cta.b64 [%0], %1;" :: "r"(fb+slot*8), "n"((unsigned)TMA_BYTES));
            asm volatile("cp.async.bulk.shared::cta.global.mbarrier::complete_tx::bytes [%0], [%1], %3, [%2];"
                :: "r"(sa + slot*TMA_BYTES), "l"(src), "r"(fb+slot*8), "n"((unsigned)TMA_BYTES) : "memory");
        }
    }
#endif

#if OP == 1 || OP == 2
    // Warps 1-3: HMMA chain (OP 1 = HMMA only; OP 2 = HMMA + TMA)
    if (wid >= 1) {
        unsigned cph = 0;
        for (int i = 0; i < ITERS; i++) {
#if OP == 2
            // Consumer: wait on full[slot], signal empty
            if (lid == 0) {
                int slot = i % DEPTH;
                unsigned t = (cph >> slot) & 1;
                unsigned p = 0;
                while (!p) asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 P, [%1], %2, 10000; selp.b32 %0, 1, 0, P; }" : "=r"(p) : "r"(fb+slot*8), "r"(t));
                cph ^= (1u << slot);
            }
            __syncwarp();
#endif
            // HMMA chain: 16x16x16 FP16 -> FP32
            #pragma unroll
            for (int k = 0; k < HMMA_PER_ITER; k++) {
                asm volatile(
                    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                    : "+r"(acc[0]), "+r"(acc[1]), "+r"(acc[2]), "+r"(acc[3])
                    : "r"(a_reg[0]), "r"(a_reg[1]), "r"(a_reg[2]), "r"(a_reg[3]),
                      "r"(b_reg[0]), "r"(b_reg[1]));
            }
#if OP == 2
            if (lid == 0) {
                int slot = i % DEPTH;
                asm volatile("mbarrier.arrive.relaxed.cta.shared::cta.b64 _, [%0];" :: "r"(eb+slot*8));
            }
            __syncwarp();
#endif
        }
    }
#endif

    __syncthreads();
    if (threadIdx.x == 0) {
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[0] = t1 - t0;
    }
    if ((int)(acc[0]^acc[1]^acc[2]^acc[3]) == seed)
        ((unsigned int*)C)[threadIdx.x + 2] = acc[0];
}
