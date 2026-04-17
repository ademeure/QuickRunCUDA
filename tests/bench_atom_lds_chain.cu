// Pure chained ATOMS/LDS latency: each op's RESULT is used as next address.
// Pre-initialize smem with valid pointer chain.

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
    // Init: smem[i] = (i + 7) & 0xFF  (creates a chain that visits all 256 cells)
    if (threadIdx.x == 0) {
        for (int i = 0; i < 256; i++) smem[i] = (i + 7) & 0xFFu;
    }
    __syncthreads();
    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned offset = 0;  // start at smem[0]

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if OP == 0
        // PURE ATOMS chain: result = atom; next addr = result * 4
        unsigned r;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base + offset * 4), "r"(0u));
        offset = r & 0xFFu;  // wrap to valid range
#elif OP == 1
        // PURE LDS chain: result = ldS; next addr = result * 4
        unsigned r;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(base + offset * 4));
        offset = r & 0xFFu;
#elif OP == 2
        // ATOMS with no return-address chain — just do ATOMS at fixed address with seed input
        unsigned r;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(seed));
        offset = r;  // use r as offset (will spread)
#elif OP == 3
        // ATOMS chain via cas (more expensive)
        unsigned r;
        asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(r) : "r"(base + offset * 4), "r"(0u), "r"(0u));
        offset = r & 0xFFu;
#elif OP == 4
        // ATOMS chain via min (no actual change since add 0)
        unsigned r;
        asm volatile("atom.shared.min.u32 %0, [%1], %2;" : "=r"(r) : "r"(base + offset * 4), "r"(0xFFFFFFFFu));
        offset = r & 0xFFu;
#elif OP == 5
        // 2 chained ATOMS (back to back, dependency through addr)
        unsigned r1, r2;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r1) : "r"(base + offset * 4), "r"(0u));
        unsigned offset2 = r1 & 0xFFu;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r2) : "r"(base + offset2 * 4), "r"(0u));
        offset = r2 & 0xFFu;
#elif OP == 6
        // 4 chained ATOMS
        unsigned r1, r2, r3, r4;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r1) : "r"(base + offset * 4), "r"(0u));
        unsigned o2 = r1 & 0xFFu;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r2) : "r"(base + o2 * 4), "r"(0u));
        unsigned o3 = r2 & 0xFFu;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r3) : "r"(base + o3 * 4), "r"(0u));
        unsigned o4 = r3 & 0xFFu;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r4) : "r"(base + o4 * 4), "r"(0u));
        offset = r4 & 0xFFu;
#elif OP == 7
        // LDS x4 chain
        unsigned r1, r2, r3, r4;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r1) : "r"(base + offset * 4));
        unsigned o2 = r1 & 0xFFu;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r2) : "r"(base + o2 * 4));
        unsigned o3 = r2 & 0xFFu;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r3) : "r"(base + o3 * 4));
        unsigned o4 = r3 & 0xFFu;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r4) : "r"(base + o4 * 4));
        offset = r4 & 0xFFu;
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = offset;
    }
}
