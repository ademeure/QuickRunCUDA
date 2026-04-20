// __restrict__ impact: does the compiler reorder/coalesce loads better?

#ifndef MODE
#define MODE 0
#endif

#if MODE == 0
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    // Without __restrict__, A and B may alias C
    C[idx] = A[idx] + B[idx];
    C[idx + 1] = A[idx + 1] + B[idx + 1];
    C[idx + 2] = A[idx + 2] + B[idx + 2];
    C[idx + 3] = A[idx + 3] + B[idx + 3];
}
#else
extern "C" __global__ void kernel(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int ITERS, int seed, int u2) {
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    // With __restrict__, compiler knows A/B/C don't alias
    C[idx] = A[idx] + B[idx];
    C[idx + 1] = A[idx + 1] + B[idx + 1];
    C[idx + 2] = A[idx + 2] + B[idx + 2];
    C[idx + 3] = A[idx + 3] + B[idx + 3];
}
#endif
