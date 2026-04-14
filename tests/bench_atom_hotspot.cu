// Hot-spot atomics — how does contention scale when multiple threads hit same address?

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef HOTSPOT_COUNT
#define HOTSPOT_COUNT 1   // # of distinct addresses (1=all threads one addr, 32=warp-unique, etc.)
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* arr = (unsigned int*)A;
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int addr_idx = tid % HOTSPOT_COUNT;

    unsigned int v = 0;
    unsigned long long addr = (unsigned long long)(arr + addr_idx);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0
            unsigned int r;
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(addr), "r"(1u));
            v ^= r;
#elif OP == 1  // .relaxed variant for less fencing
            unsigned int r;
            asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(addr), "r"(1u));
            v ^= r;
#elif OP == 2  // CAS on hotspot
            unsigned int r;
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r) : "l"(addr), "r"(0u), "r"(1u));
            v ^= r;
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[tid] = v;
}
