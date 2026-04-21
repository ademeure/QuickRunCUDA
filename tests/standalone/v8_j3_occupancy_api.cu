// V8 J3: cudaOccupancyMaxPotentialBlockSize API check
// Test if CUDA can recommend optimal block size automatically
#include <cuda_runtime.h>
#include <cstdio>

__global__ void big_regs(float* A, int n) {
    float v[32];
    #pragma unroll
    for (int i = 0; i < 32; i++) v[i] = (float)threadIdx.x + i;
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < 32; i++) sum += v[i];
    if (sum == 1.234567e-30f) A[0] = sum;
}

__global__ void small_regs(float* A, int n) {
    float v = (float)threadIdx.x;
    if (v == 1.234567e-30f) A[0] = v;
}

int main() {
    cudaSetDevice(0);

    int min_grid_size, block_size;

    // Auto-pick for big_regs (high register usage)
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, big_regs, 0, 0);
    printf("big_regs (32 floats):\n");
    printf("  Recommended block size: %d, min grid: %d\n", block_size, min_grid_size);

    // Auto-pick for small_regs
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, small_regs, 0, 0);
    printf("small_regs (1 float):\n");
    printf("  Recommended block size: %d, min grid: %d\n", block_size, min_grid_size);

    // For each, also query active blocks per SM at common block sizes
    for (int bs : {32, 128, 256, 512, 1024}) {
        int num_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&num_blocks, big_regs, bs, 0);
        printf("big_regs at %d threads: %d blocks/SM (= %d threads/SM)\n",
               bs, num_blocks, bs * num_blocks);
    }

    return 0;
}
