// Clearly-scoped L1/L2/DRAM bandwidth test with known working-set sizes.

#ifndef UNROLL
#define UNROLL 4
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef WINDOW_KB
#define WINDOW_KB 64
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int elems_in_window = (WINDOW_KB * 1024) / 4;  // u32 elements
    unsigned int acc0=0,acc1=0,acc2=0,acc3=0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Each thread touches the whole window cyclically (sequential, coalesced)
            unsigned int base_idx = ((threadIdx.x + j * blockDim.x + i * UNROLL * blockDim.x) * 4) % elems_in_window;
            unsigned long long addr = (unsigned long long)(A + base_idx);
            unsigned int x0,x1,x2,x3;
            asm volatile("ld.global.v4.u32 {%0,%1,%2,%3}, [%4];" : "=r"(x0),"=r"(x1),"=r"(x2),"=r"(x3) : "l"(addr));
            acc0^=x0; acc1^=x1; acc2^=x2; acc3^=x3;
        }
    }
    unsigned int r = acc0^acc1^acc2^acc3;
    if ((int)r == seed) ((unsigned int*)C)[tid] = r;
}
