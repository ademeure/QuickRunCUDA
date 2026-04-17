// Reserved 1 KB shared memory: probe per-block visible SHMEM
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void probe_smem(int *out, int request_size) {
    extern __shared__ char smem[];
    int tid = threadIdx.x;
    if (tid == 0) {
        // Write a unique pattern to each byte from offset 0 to (request_size + 8KB margin)
        // and check if writes to higher offsets succeed
        out[blockIdx.x] = request_size;
    }
    // Just touch enough of smem to keep it allocated
    if (tid < request_size / 4) {
        ((int*)smem)[tid] = tid;
    }
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));

    cudaFuncAttributes attr;
    cudaFuncGetAttributes(&attr, probe_smem);
    printf("# B300 reserved SHMEM probe\n");
    printf("# Initial maxDynamicSharedSize: %d\n", attr.maxDynamicSharedSizeBytes);

    // Set max dynamic SHMEM to opt-in max
    int max_smem;
    cudaDeviceGetAttribute(&max_smem, cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);
    printf("# Device max opt-in SHMEM: %d B\n", max_smem);
    cudaFuncSetAttribute(probe_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, max_smem);

    int reserved;
    cudaDeviceGetAttribute(&reserved, cudaDevAttrReservedSharedMemoryPerBlock, 0);
    printf("# Reserved SHMEM per block: %d B\n\n", reserved);

    // Try various dynamic SHMEM sizes
    printf("# Testing various dynamic SHMEM allocation sizes:\n");
    printf("# %-15s %-15s %-15s\n", "request_KB", "result", "max_blocks/SM");

    for (int kb : {0, 1, 16, 56, 64, 128, 200, 220, 224, 226, 227, 230, 240}) {
        int req = kb * 1024;
        cudaError_t err = cudaSuccess;
        int max_blocks = -1;

        // Try to launch
        probe_smem<<<1, 32, req>>>(d_out, req);
        err = cudaDeviceSynchronize();

        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, probe_smem, 32, req);

        if (err == cudaSuccess) {
            printf("  %-15d OK              %-15d\n", kb, max_blocks);
        } else {
            printf("  %-15d ERR: %s\n", kb, cudaGetErrorString(err));
            cudaGetLastError();  // clear
        }
    }

    return 0;
}
