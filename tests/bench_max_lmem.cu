// Max LMEM (local memory) per thread test
#ifndef LMEM_DWORDS
#define LMEM_DWORDS 256
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // Force lmem with large array beyond register file
    unsigned int big[LMEM_DWORDS];
    int idx = (threadIdx.x + (unsigned)u2) & (LMEM_DWORDS - 1);
    #pragma unroll 1
    for (int i = 0; i < LMEM_DWORDS; i++) big[i] = (unsigned)i + idx;
    __syncthreads();
    unsigned int x = 0;
    #pragma unroll 1
    for (int i = 0; i < LMEM_DWORDS; i++) x ^= big[(i + idx) & (LMEM_DWORDS - 1)];
    if (x == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = x;
}
