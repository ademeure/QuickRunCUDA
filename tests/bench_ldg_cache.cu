// LDG cache-hint variants with proper dep chain so DCE can't hide cost.

#ifndef UNROLL
#define UNROLL 8
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* arr = (unsigned int*)A;
    // Each thread starts at a random offset and chases through a warp-coalescing
    // access pattern so we measure actual DRAM bandwidth.
    unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned int r;
            unsigned long long addr = (unsigned long long)(arr + (idx & 0x7FFFFF));  // 32MB window
#if OP == 0  // ld.global.u32 (default cache policy = .ca)
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 1  // ld.global.ca (cache all levels)
            asm volatile("ld.global.ca.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 2  // ld.global.cg (cache L2 only, bypass L1)
            asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 3  // ld.global.cs (streaming, evict-first)
            asm volatile("ld.global.cs.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 4  // ld.global.lu (last-use, don't cache)
            asm volatile("ld.global.lu.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 5  // ld.global.nc (non-coherent, read-only cache)
            asm volatile("ld.global.nc.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 6  // ld.volatile.global.u32
            asm volatile("ld.volatile.global.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#endif
            v ^= r;
            idx = idx * 1103515245u + 12345u;  // LCG — data-dep address evolution
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
