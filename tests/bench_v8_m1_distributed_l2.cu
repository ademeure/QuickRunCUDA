// V8 M1: Distributed L2 cache via setAccessPolicyWindow on stream
// Test if hitRatio tuning changes effective cache use for cluster-wide workloads
#include <cuda_runtime.h>

extern "C" __global__ __launch_bounds__(256, 1)
__cluster_dims__(8, 1, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int MASK = (16 * 1024 * 1024) - 1;

    float sum = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned int idx = (gtid * 37 + i) & MASK;
        sum += A[idx];
    }

    if (sum == 1.234567e-30f) C[gtid] = sum;
}
