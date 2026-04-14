// DRAM bandwidth for different access patterns.

#ifndef UNROLL
#define UNROLL 4
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef PATTERN
#define PATTERN 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;
    unsigned int acc0=0,acc1=0,acc2=0,acc3=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long off;
#if PATTERN == 0  // Sequential coalesced (stride 1 v4 across threads in warp)
            off = (unsigned long long)tid * 16 + (unsigned long long)(i + j) * 16 * total_threads;
#elif PATTERN == 1  // Stride 2 (alternate lanes skipped)
            off = (unsigned long long)tid * 32 + (unsigned long long)(i + j) * 32 * total_threads;
#elif PATTERN == 2  // Stride 4 (4 lanes skipped)
            off = (unsigned long long)tid * 64 + (unsigned long long)(i + j) * 64 * total_threads;
#elif PATTERN == 3  // Stride 8 (cacheline skip)
            off = (unsigned long long)tid * 128 + (unsigned long long)(i + j) * 128 * total_threads;
#elif PATTERN == 4  // Stride 16 (2 cachelines)
            off = (unsigned long long)tid * 256 + (unsigned long long)(i + j) * 256 * total_threads;
#elif PATTERN == 5  // Random (LCG, 32MB window)
            unsigned int lcg = (tid + j*7 + i*11) * 1103515245u + 12345u;
            off = (lcg & 0x1FFFFFFu);
#elif PATTERN == 6  // Strided read by block (each block contiguous but blocks overlap)
            off = (unsigned long long)threadIdx.x * 16 + (unsigned long long)blockIdx.x * 4096 + (unsigned long long)(i+j) * 16 * blockDim.x;
#endif
            unsigned long long addr = (unsigned long long)A + (off & 0x1FFFFFFFu);  // 512 MB cap
            unsigned int x0,x1,x2,x3;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    unsigned int r = acc0^acc1^acc2^acc3;
    if ((int)r == seed) ((unsigned int*)C)[tid] = r;
}
