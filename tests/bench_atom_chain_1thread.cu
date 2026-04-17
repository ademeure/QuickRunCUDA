#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (threadIdx.x == 0) {
        for (int i = 0; i < 256; i++) smem[i] = (i + 7) & 0xFFu;
    }
    __syncthreads();
#if OP == 0 || OP == 2
    if (threadIdx.x != 0) return;  // PURE 1 thread per SM
#endif
    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned offset = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < 1024; i++) {
#if OP == 0
        // 1 thread, ATOMS chain
        unsigned r;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base + offset * 4), "r"(0u));
        offset = r & 0xFFu;
#elif OP == 1
        // 32 threads (full warp), each its own chain (different starting offset per lane)
        unsigned r;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base + ((offset + threadIdx.x) & 0xFFu) * 4), "r"(0u));
        offset = r & 0xFFu;
#elif OP == 2
        // 1 thread, LDS chain
        unsigned r;
        asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(base + offset * 4));
        offset = r & 0xFFu;
#elif OP == 3
        // 32 threads, all same address ATOMS (warp coalesce)
        unsigned r;
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base + offset * 4), "r"(0u));
        offset = r & 0xFFu;
#endif
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = offset;
    }
}
