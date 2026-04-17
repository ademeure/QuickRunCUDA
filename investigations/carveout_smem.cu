// Effect of cudaFuncAttributePreferredSharedMemoryCarveout on occupancy
#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel_use_smem(float *out, int N) {
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    smem[tid] = (float)tid;
    __syncthreads();
    float sum = 0;
    for (int i = 0; i < N; i++) sum += smem[(tid + i) & (blockDim.x - 1)];
    if (sum < -1e30f) out[blockIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, kernel_use_smem);
    printf("# B300 kernel attributes:\n");
    printf("  numRegs:               %d\n", attr.numRegs);
    printf("  maxThreadsPerBlock:    %d\n", attr.maxThreadsPerBlock);
    printf("  sharedSizeBytes:       %zd\n", attr.sharedSizeBytes);
    printf("  maxDynamicSharedSize:  %d\n", attr.maxDynamicSharedSizeBytes);
    printf("  preferredShmemCarveout: %d\n\n", attr.preferredShmemCarveout);

    int dev_max_smem;
    cudaDeviceGetAttribute(&dev_max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    printf("# Device max opt-in shared mem per block: %d B = %d KB\n\n",
           dev_max_smem, dev_max_smem/1024);

    // Set max dynamic SHMEM
    int requested = dev_max_smem;
    cudaError_t err = cudaFuncSetAttribute(kernel_use_smem,
        cudaFuncAttributeMaxDynamicSharedMemorySize, requested);
    printf("  cudaFuncSetAttribute max=%d KB: %s\n",
           requested/1024, cudaGetErrorString(err));

    // Carveout values: 0 = default, 100 = 100% smem
    printf("\n# Trying various carveout values, occupancy at 256 thr/block:\n");
    printf("# %-15s %-15s %-15s\n", "carveout%", "smem_avail", "blocks/SM");

    int cs[] = {0, 25, 50, 75, 100, cudaSharedmemCarveoutDefault, cudaSharedmemCarveoutMaxL1, cudaSharedmemCarveoutMaxShared}; for (int c : cs) {
        cudaFuncSetAttribute(kernel_use_smem,
            cudaFuncAttributePreferredSharedMemoryCarveout, c);

        int max_blocks;
        // Test with 32 KB SHMEM per block
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel_use_smem, 256, 32*1024);

        const char *desc = "";
        if (c == cudaSharedmemCarveoutDefault) desc = " (Default)";
        else if (c == cudaSharedmemCarveoutMaxL1) desc = " (MaxL1)";
        else if (c == cudaSharedmemCarveoutMaxShared) desc = " (MaxShared)";
        printf("  c=%-3d%-12s    32KB asked, %d blocks/SM\n", c, desc, max_blocks);
    }

    // Test how SHMEM size affects occupancy
    printf("\n# 256 thr/block, vary SHMEM/block:\n");
    printf("# %-15s %-15s\n", "smem_per_block_KB", "blocks/SM");

    cudaFuncSetAttribute(kernel_use_smem,
        cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutDefault);

    for (int kb : {0, 8, 16, 32, 48, 56, 80, 100, 128, 160, 200, 220}) {
        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, kernel_use_smem, 256, kb*1024);
        printf("  %-15d %d\n", kb, max_blocks);
    }

    return 0;
}
