// Quantify LDS / STS / ATOMS bank-conflict cost curve.
// STRIDE controls the addressing gap.

#ifndef UNROLL
#define UNROLL 16
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
#ifndef STRIDE
#define STRIDE 1     // lane t hits word smem[t*STRIDE]. STRIDE=1 no conflict.
#endif

extern __shared__ unsigned int smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int tid = threadIdx.x;
    unsigned int base = (unsigned)__cvta_generic_to_shared(&smem[tid * STRIDE]);
    if (tid < 8192/4) smem[tid] = tid;
    __syncthreads();

    unsigned int v = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#if OP == 0  // ld.shared.u32
            unsigned int r;
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(r) : "r"(base));
            v ^= r;
#elif OP == 1  // st.shared.u32
            asm volatile("st.shared.u32 [%0], %1;" :: "r"(base), "r"(v + j));
#elif OP == 2  // atom.shared.add.u32
            unsigned int r;
            asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(1u));
            v ^= r;
#endif
        }
    }
    if ((int)v == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = v;
}
