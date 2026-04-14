// LDS bandwidth vs number of active warps per block.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = threadIdx.x;
    if (tid < 512) smem[tid] = tid;
    __syncthreads();

    unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[tid]);
    unsigned int v = tid;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // data-dep chain so no DCE
            unsigned int r;
            unsigned int addr = base + (v & 0x1F) * 4;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(addr));
            v = r + 1;
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + tid] = v;
}
