// Can we fit 4 blocks × 57 KiB shmem on a B300 SM?
// Try every trick possible
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define KB(x) ((x) * 1024)

extern "C" __global__ void k_57k_dyn() {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    buf[tid * 100] = (char)tid;
    __syncthreads();
    if (tid == 0) printf("57k_dyn: blockIdx=%d, ok\n", blockIdx.x);
}

// Static 57 KB
extern "C" __global__ void k_57k_static() {
    __shared__ char buf[KB(57)];
    int tid = threadIdx.x;
    buf[tid * 100] = (char)tid;
    __syncthreads();
    if (tid == 0) printf("57k_static: blockIdx=%d ok\n", blockIdx.x);
}

// Half static, half dyn
extern "C" __global__ void k_57k_split() {
    __shared__ char st[KB(28)];
    extern __shared__ char dyn[];
    int tid = threadIdx.x;
    st[tid % KB(28)] = (char)tid;
    if (tid < KB(29)) dyn[tid] = 1;
    __syncthreads();
    if (tid == 0) printf("57k_split: blockIdx=%d ok\n", blockIdx.x);
}

void test_kernel(const char *name, const void *func, int dyn_bytes, int static_bytes) {
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    // For dynamic kernels, set opt-in
    if (dyn_bytes > 0) {
        cudaError_t r = cudaFuncSetAttribute((void*)func,
                                             cudaFuncAttributeMaxDynamicSharedMemorySize, dyn_bytes);
        if (r != cudaSuccess) {
            printf("  %s: cudaFuncSetAttribute(MaxDynShmem=%d) FAILED: %s\n",
                   name, dyn_bytes, cudaGetErrorString(r));
            return;
        }
    }

    // Try various PreferredSharedMemoryCarveout values
    int carveouts[] = {-1, 0, 25, 50, 75, 100};
    int total_per_block = static_bytes + dyn_bytes + 1024;  // include reserved

    printf("\n# Testing %s (static=%d, dyn=%d, total/block w/ reserved=%d)\n",
           name, static_bytes, dyn_bytes, total_per_block);

    for (int carve : carveouts) {
        if (carve >= 0) {
            cudaFuncSetAttribute((void*)func, cudaFuncAttributePreferredSharedMemoryCarveout, carve);
        }

        int max_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks, (void*)func, 32, dyn_bytes);

        // Also query function attrs
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, (void*)func);

        printf("  carveout=%-3d: max_blocks/SM=%d (sharedSize=%zu, maxDynShmem=%d)\n",
               carve, max_blocks, fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes);

        if (max_blocks >= 4) {
            // Try to actually launch 4 blocks
            cudaError_t r;
            cudaGetLastError();
            void *args[] = {};
            cudaLaunchKernel((void*)func, dim3(4), dim3(32), args, dyn_bytes, 0);
            r = cudaGetLastError();
            if (r == cudaSuccess) {
                cudaDeviceSynchronize();
                printf("    → Launch 4 blocks: SUCCESS!\n");
            } else {
                printf("    → Launch 4 blocks: %s\n", cudaGetErrorString(r));
            }
        }
    }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

    printf("# B300 SM shmem: %zu bytes (%.1f KB)\n",
           prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor/1024.0);
    printf("# Per-block opt-in max: %zu (%.1f KB)\n",
           prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin/1024.0);
    printf("# Reserved per block: %zu\n", prop.reservedSharedMemPerBlock);
    printf("# Theoretical max if 4 blocks of N KiB: 4*(N*1024 + 1024) ≤ %zu\n",
           prop.sharedMemPerMultiprocessor);
    printf("# → max N per-block-user = (%zu/4 - 1024) = %zu = %.1f KiB\n",
           prop.sharedMemPerMultiprocessor,
           prop.sharedMemPerMultiprocessor/4 - 1024,
           (prop.sharedMemPerMultiprocessor/4.0 - 1024)/1024.0);

    // 57 KiB = 58368 bytes
    test_kernel("k_57k_dyn", (void*)k_57k_dyn, KB(57), 0);
    test_kernel("k_57k_static", (void*)k_57k_static, 0, KB(57));
    test_kernel("k_57k_split", (void*)k_57k_split, KB(29), KB(28));

    // What's the MAX shmem we can have to fit 4 blocks?
    printf("\n## What's the MAX dyn shmem to fit 4 blocks?\n");
    int test_sizes[] = {KB(48), KB(50), KB(52), KB(55), KB(56), KB(57), KB(58)};
    for (int sz : test_sizes) {
        cudaFuncSetAttribute((void*)k_57k_dyn, cudaFuncAttributeMaxDynamicSharedMemorySize, sz);
        int blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)k_57k_dyn, 32, sz);
        printf("  dyn=%-7d (%.1f KB): max %d blocks/SM\n", sz, sz/1024.0, blocks);
    }

    return 0;
}
