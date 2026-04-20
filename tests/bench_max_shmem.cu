// Max dynamic SHMEM allocation test.
extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    extern __shared__ float smem[];
    smem[threadIdx.x] = (float)threadIdx.x + (float)u2;
    __syncthreads();
    if (threadIdx.x == 0) C[blockIdx.x] = smem[0];
}
