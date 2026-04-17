extern "C" __global__ __launch_bounds__(1, 1) void kernel(unsigned* A, unsigned* B, unsigned* C, int ITERS, int seed, int u2) {
    unsigned v = seed;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned r;
#if OP == 0  // global atom.add (self-dep via return)
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"((unsigned long long)A), "r"(v));
        v = r + 1;
#elif OP == 1  // global atom.cas
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r) : "l"((unsigned long long)A), "r"(v), "r"(v+1));
        v = r + 1;
#elif OP == 2  // global atom.relaxed.sys
        asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"((unsigned long long)A), "r"(v));
        v = r + 1;
#elif OP == 3  // shared atom.add
        __shared__ unsigned smem_loc;
        unsigned addr = (unsigned)__cvta_generic_to_shared(&smem_loc);
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(addr), "r"(v));
        v = r + 1;
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    ((unsigned long long*)C)[0] = t1 - t0;
    C[2] = v;  // unconditional use
}
