// Clarify "atom.shared.add 28 cy pure" — measure both same-addr throughput
// (no dep chain) and pipelined different-addr throughput.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (threadIdx.x == 0) for (int i = 0; i < 256; i++) smem[i] = i;
    __syncthreads();
    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        // Single atom with chained dep through return value (latency-bound)
        unsigned v = (unsigned)seed + i;
        unsigned r;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(v));
        v = r + 1;
        if (r == 0xDEADBEEF) ((unsigned*)C)[1] = v;  // keep r live without dep chain to next inst
#elif OP == 1
        // 4 INDEPENDENT atoms to SAME address (no chain, no dep between atoms)
        unsigned r0, r1, r2, r3;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r0) : "r"(base), "r"(seed+0));
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r1) : "r"(base), "r"(seed+1));
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r2) : "r"(base), "r"(seed+2));
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r3) : "r"(base), "r"(seed+3));
        if ((r0|r1|r2|r3) == 0xDEADBEEF) ((unsigned*)C)[1] = r0+r1+r2+r3;
#elif OP == 2
        // 4 INDEPENDENT atoms to DIFFERENT addresses
        unsigned r0, r1, r2, r3;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r0) : "r"(base+0),  "r"(seed+0));
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r1) : "r"(base+4),  "r"(seed+1));
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r2) : "r"(base+8),  "r"(seed+2));
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r3) : "r"(base+12), "r"(seed+3));
        if ((r0|r1|r2|r3) == 0xDEADBEEF) ((unsigned*)C)[1] = r0+r1+r2+r3;
#elif OP == 3
        // 8 indep atoms different-addr — push throughput
        unsigned r[8];
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r[k]) : "r"(base+k*4), "r"(seed+k));
        }
        unsigned sum = 0;
        #pragma unroll
        for (int k = 0; k < 8; k++) sum |= r[k];
        if (sum == 0xDEADBEEF) ((unsigned*)C)[1] = sum;
#elif OP == 4
        // red.shared (no return) — fastest write path
        asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base), "r"(seed+i) : "memory");
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
    }
}
