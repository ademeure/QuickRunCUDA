// Proper LDS bank-conflict curve — loaded values feed a dependency chain
// that the compiler cannot eliminate.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef STRIDE
#define STRIDE 1
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = threadIdx.x;
    // Initialize SMEM with unique values based on tid so compiler can't fold
    if (tid < BLOCK_SIZE * STRIDE) smem[tid] = tid * 0x5DEECE66Du;
    __syncthreads();

    unsigned int v = tid;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Use v to compute dynamic address so every LDS address is
            // data-dependent and compiler can't predict.
            unsigned int idx = (v ^ (tid * STRIDE + j)) & (BLOCK_SIZE * STRIDE - 1);
            unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[idx]);
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(base));
            v = r;  // full dep chain
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
