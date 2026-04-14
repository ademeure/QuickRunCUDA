// DRAM row-buffer hit effect: sequential-within-page vs random-across-pages.

#ifndef UNROLL
#define UNROLL 4
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_threads = gridDim.x * blockDim.x;
    unsigned int acc0=0, acc1=0, acc2=0, acc3=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long off;
#if MODE == 0  // Pure sequential (best case)
            off = (unsigned long long)tid * 16 + (unsigned long long)(i + j) * 16 * total_threads;
#elif MODE == 1  // Stride 4KB (DRAM page boundary for HBM)
            off = (unsigned long long)tid * 16 + (unsigned long long)(i + j) * 4096 * total_threads;
#elif MODE == 2  // Random per iteration within 1GB window
            unsigned int lcg = (tid + (i+j) * 13) * 1103515245u + 12345u;
            off = ((unsigned long long)lcg & 0x3FFFFFFFu);
#elif MODE == 3  // Random per-block (each block has random base)
            unsigned int lcg = (blockIdx.x + (i+j) * 13) * 1103515245u;
            off = (unsigned long long)(lcg & 0x3FFFFFFFu) + (unsigned long long)threadIdx.x * 16;
#elif MODE == 4  // Stride 64 (per-warp cacheline stride)
            off = (unsigned long long)tid * 64 + (unsigned long long)(i + j) * 64 * total_threads;
#endif
            unsigned long long addr = (unsigned long long)A + (off & 0x0FFFFFFFu);  // 256MB wrap
            unsigned int x0,x1,x2,x3;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    unsigned int r = acc0^acc1^acc2^acc3;
    if ((int)r == seed) ((unsigned int*)C)[tid] = r;
}
