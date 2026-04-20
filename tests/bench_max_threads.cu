extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x == 0) C[blockIdx.x] = (float)blockDim.x;
}
