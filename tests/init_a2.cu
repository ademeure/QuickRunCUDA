// Init kernel for A2: zero out A buffer
extern "C" __global__ void init(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    A[i] = 0.0f;
}
