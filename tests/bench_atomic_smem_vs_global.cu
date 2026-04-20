// SMEM atomic throughput vs global atomic
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[256];
    if (threadIdx.x < 256) smem[threadIdx.x] = 0;
    __syncthreads();

    int* gp = (int*)A;
    int idx = threadIdx.x;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // SMEM atomic - distinct addresses per thread
        atomicAdd(&smem[idx], 1);
#elif MODE == 1
        // Global atomic - distinct addresses per thread
        atomicAdd(gp + idx + blockIdx.x * 256, 1);
#elif MODE == 2
        // SMEM atomic - all threads same address (heavy contention)
        atomicAdd(&smem[0], 1);
#elif MODE == 3
        // Global atomic - all threads same address
        atomicAdd(gp + blockIdx.x, 1);
#endif
    }

    __syncthreads();
    if (threadIdx.x == 0) C[blockIdx.x] = (float)smem[0];
}
