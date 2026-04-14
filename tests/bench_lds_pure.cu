// Pure LDS throughput
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef N_CHAINS
#define N_CHAINS 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    __shared__ unsigned int smem[4096];
    if (threadIdx.x < 4096) smem[threadIdx.x] = threadIdx.x * 0x12345u;
    __syncthreads();
    unsigned int macc[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) macc[k] = 0;
    unsigned int idx0 = threadIdx.x & 1023;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int v;
                unsigned int idx = (idx0 + k*7 + j*3) & 1023;
                asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "l"((size_t)(smem + idx)));
                macc[k] ^= v;
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= macc[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
