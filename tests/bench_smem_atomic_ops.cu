// SMEM atomic op variants
#include <cuda_fp16.h>
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[256];
    __shared__ __half2 hsmem[256];
    if (threadIdx.x < 256) {
        smem[threadIdx.x] = 0;
        hsmem[threadIdx.x] = __half2{};
    }
    __syncthreads();

    int idx = threadIdx.x;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        atomicAdd(&smem[idx], 1u);
#elif MODE == 1
        atomicMin(&smem[idx], (unsigned)i);
#elif MODE == 2
        atomicMax(&smem[idx], (unsigned)i);
#elif MODE == 3
        atomicCAS(&smem[idx], 0, (unsigned)i);
#elif MODE == 4
        atomicAdd(&hsmem[idx], __float2half2_rn(1.0f));
#endif
    }
    __syncthreads();
    if (threadIdx.x == 0) C[blockIdx.x] = (float)smem[0];
}
