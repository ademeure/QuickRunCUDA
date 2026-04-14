// L2 eviction-priority hints (sm_80+).

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
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int v = 0;
    unsigned int idx = tid;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned int r;
            unsigned long long addr = (unsigned long long)(A + (idx & 0x1FFFFFF));  // 128 MB window
#if OP == 0  // default
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 1  // L2::evict_last (prioritize keeping in L2)
            asm volatile("ld.global.L2::evict_last.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 2  // L2::evict_first (deprioritize — evict soon)
            asm volatile("ld.global.L2::evict_first.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 3  // L2::evict_normal (default)
            asm volatile("ld.global.L2::evict_normal.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 4  // L2::evict_unchanged
            asm volatile("ld.global.L2::evict_unchanged.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#elif OP == 5  // L2::no_allocate (bypass L2)
            asm volatile("ld.global.L2::no_allocate.u32 %0, [%1];" : "=r"(r) : "l"(addr));
#endif
            v ^= r;
            idx = idx * 1103515245u + 12345u;
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[tid] = v;
}
