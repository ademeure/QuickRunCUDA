// CREATIVE attempts to fit 4 blocks × 57 KiB shmem on a B300 SM
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

#define KB(x) ((x) * 1024)
#define TARGET_BYTES (57 * 1024)  // 58368 bytes

// REAL kernel that actually uses 57 KiB - prevents DCE
extern "C" __global__ void k_real_dyn() {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    // Touch every byte of buf (forces real allocation)
    for (int i = tid; i < TARGET_BYTES; i += blockDim.x)
        buf[i] = (char)(tid ^ bid);
    __syncthreads();
    // Read back to register, write to global to prevent DCE
    extern __shared__ char buf_alias[];  // alias
    int sum = 0;
    for (int i = tid; i < TARGET_BYTES; i += blockDim.x)
        sum += (int)buf_alias[i];
    if (tid == 0) {
        // Use atomic to prevent dead-store elimination
        atomicAdd((int*)((long)buf + (TARGET_BYTES - 4)), sum);
    }
}

extern "C" __global__ void k_real_static() {
    __shared__ volatile char buf[TARGET_BYTES];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    for (int i = tid; i < TARGET_BYTES; i += blockDim.x)
        buf[i] = (char)(tid ^ bid);
    __syncthreads();
    int sum = 0;
    for (int i = tid; i < TARGET_BYTES; i += blockDim.x)
        sum += (int)buf[i];
    if (tid == 0) {
        atomicAdd((int*)&buf[TARGET_BYTES - 4], sum);
    }
}

// Cluster version
extern "C" __global__ void __cluster_dims__(4,1,1) k_cluster_real() {
    extern __shared__ char buf[];
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    for (int i = tid; i < TARGET_BYTES; i += blockDim.x)
        buf[i] = (char)(tid ^ bid);
    __syncthreads();
    int sum = 0;
    for (int i = tid; i < TARGET_BYTES; i += blockDim.x)
        sum += (int)buf[i];
    if (tid == 0) {
        atomicAdd((int*)((long)buf + (TARGET_BYTES - 4)), sum);
    }
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));

    printf("# B300 creative test: 4 blocks × %d bytes (%.1f KiB) shmem on one SM\n\n",
           TARGET_BYTES, TARGET_BYTES/1024.0);
    printf("# SM total: %zu, opt-in: %zu, reserved: %zu\n\n",
           prop.sharedMemPerMultiprocessor, prop.sharedMemPerBlockOptin,
           prop.reservedSharedMemPerBlock);

    // TEST 1: Pure dynamic shmem with REAL usage (no DCE)
    printf("## TEST 1: pure dyn shmem, real usage\n");
    {
        cudaError_t r = cudaFuncSetAttribute((void*)k_real_dyn,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              TARGET_BYTES);
        printf("  cudaFuncSetAttribute(MaxDynShmem=%d): %s\n",
               TARGET_BYTES, r == cudaSuccess ? "OK" : cudaGetErrorString(r));
        int blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)k_real_dyn, 256, TARGET_BYTES);
        printf("  Max blocks/SM (occupancy API): %d\n", blocks);

        // Try launching 4 blocks on 1 SM (using <<<4, ...>>> means 4 blocks total)
        // Actually to force same-SM, we need persistent... use grid = 4 blocks total
        for (int g : {1, 2, 3, 4, 5}) {
            cudaGetLastError();
            k_real_dyn<<<g, 256, TARGET_BYTES>>>();
            cudaDeviceSynchronize();
            cudaError_t r = cudaGetLastError();
            printf("    Launch <<<%d,256, %d>>>: %s\n", g, TARGET_BYTES,
                   r == cudaSuccess ? "OK" : cudaGetErrorString(r));
        }
    }

    // TEST 2: Static shmem with REAL usage
    printf("\n## TEST 2: static %d-byte shmem, real usage\n", TARGET_BYTES);
    {
        cudaFuncAttributes fa;
        cudaFuncGetAttributes(&fa, (void*)k_real_static);
        printf("  k_real_static: sharedSize=%zu (vs target %d)\n", fa.sharedSizeBytes, TARGET_BYTES);
        int blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)k_real_static, 256, 0);
        printf("  Max blocks/SM (occupancy API): %d\n", blocks);

        for (int g : {1, 2, 3, 4, 5}) {
            cudaGetLastError();
            k_real_static<<<g, 256>>>();
            cudaDeviceSynchronize();
            cudaError_t r = cudaGetLastError();
            printf("    Launch <<<%d,256>>>: %s\n", g,
                   r == cudaSuccess ? "OK" : cudaGetErrorString(r));
        }
    }

    // TEST 3: Cluster of 4 with same dyn shmem - maybe per-cluster reservation?
    printf("\n## TEST 3: cluster=4, each block %d bytes dyn\n", TARGET_BYTES);
    {
        cudaError_t r = cudaFuncSetAttribute((void*)k_cluster_real,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize,
                                              TARGET_BYTES);
        printf("  Set attr: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

        for (int g : {4, 8, 16}) {
            cudaGetLastError();
            k_cluster_real<<<g, 256, TARGET_BYTES>>>();
            cudaDeviceSynchronize();
            cudaError_t r = cudaGetLastError();
            printf("    Launch <<<%d,256, %d>>> cluster_dim=4: %s\n", g, TARGET_BYTES,
                   r == cudaSuccess ? "OK" : cudaGetErrorString(r));
        }
    }

    // TEST 4: Check if `cudaSharedMemoryMode` attribute exists/helps
    printf("\n## TEST 4: cudaSharedMemoryMode (id=18)\n");
    {
        // The enum has cudaSharedMemoryMode but I'm not sure what values it accepts
        // Try setting to non-default
        cudaLaunchAttribute attr;
        attr.id = (cudaLaunchAttributeID)18; // SharedMemoryMode
        attr.val.sharedMemoryMode = (cudaSharedMemoryMode)1;  // try value 1
        cudaLaunchConfig_t cfg = {dim3(4), dim3(256), TARGET_BYTES, 0, &attr, 1};
        void *args[] = {};

        cudaError_t r = cudaLaunchKernelExC(&cfg, (void*)k_real_dyn, args);
        cudaDeviceSynchronize();
        printf("  SharedMemoryMode=1: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));

        attr.val.sharedMemoryMode = (cudaSharedMemoryMode)0;
        r = cudaLaunchKernelExC(&cfg, (void*)k_real_dyn, args);
        cudaDeviceSynchronize();
        printf("  SharedMemoryMode=0: %s\n", r == cudaSuccess ? "OK" : cudaGetErrorString(r));
    }

    // TEST 5: How much can we ACTUALLY fit?
    printf("\n## TEST 5: Sweep dyn shmem to find max for 4-block occupancy\n");
    int sweep[] = {KB(48), KB(50), KB(52), KB(55), KB(56), 57000, 57344, 57500, 57600, 57700, 57800, 57900, 58000, 58200, KB(57)};
    for (int sz : sweep) {
        cudaError_t r = cudaFuncSetAttribute((void*)k_real_dyn,
                                              cudaFuncAttributeMaxDynamicSharedMemorySize, sz);
        if (r != cudaSuccess) {
            printf("  size=%-7d (%6.2f KiB): set attr fail: %s\n", sz, sz/1024.0, cudaGetErrorString(r));
            continue;
        }
        int blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, (void*)k_real_dyn, 256, sz);
        printf("  size=%-7d (%6.2f KiB): max %d blocks/SM\n", sz, sz/1024.0, blocks);
    }

    return 0;
}
