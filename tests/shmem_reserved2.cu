// Cleaner driver-reserved 1 KiB shmem investigation
#include <cuda_runtime.h>
#include <cstdio>
#include <cooperative_groups.h>

#define CK(c) do { cudaError_t e=(c); if(e!=cudaSuccess){ \
    fprintf(stderr,"CUDA %s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)

namespace cg = cooperative_groups;

// Variant 1: simple kernel, no special features
extern "C" __global__ void k_simple() {
    if (threadIdx.x == 0) printf("simple: bid=%d\n", blockIdx.x);
}

// Variant 2: uses cooperative_groups (this_grid)
extern "C" __global__ void k_coop() {
    auto grid = cg::this_grid();
    grid.sync();
    if (threadIdx.x == 0) printf("coop: bid=%d\n", blockIdx.x);
}

// Variant 3: uses __syncthreads
extern "C" __global__ void k_sync() {
    __syncthreads();
    if (threadIdx.x == 0) printf("sync: bid=%d\n", blockIdx.x);
}

// Variant 4: uses cluster
extern "C" __global__ void __cluster_dims__(2,1,1) k_cluster() {
    auto cluster = cg::this_cluster();
    cluster.sync();
    if (threadIdx.x == 0) printf("cluster: bid=%d cluster_rank=%d\n", blockIdx.x, cluster.block_rank());
}

// Variant 5: declares dyn shmem
extern "C" __global__ void k_dyn() {
    extern __shared__ float dyn_buf[];
    if (threadIdx.x == 0) {
        dyn_buf[0] = 1.0f;
        printf("dyn: dyn_buf[0]=%.1f\n", dyn_buf[0]);
    }
}

// Variant 6: declares static shmem
extern "C" __global__ void k_static() {
    __shared__ float st[256];
    st[threadIdx.x % 256] = (float)threadIdx.x;
    __syncthreads();
    if (threadIdx.x == 0) printf("static: st[0]=%.1f\n", st[0]);
}

int main() {
    CK(cudaSetDevice(0));
    cudaDeviceProp prop; CK(cudaGetDeviceProperties(&prop, 0));
    printf("# Driver-reserved shmem: %zu bytes\n\n", prop.reservedSharedMemPerBlock);

    cudaFuncAttributes fa;
    void *kernels[] = {
        (void*)k_simple, (void*)k_coop, (void*)k_sync,
        (void*)k_cluster, (void*)k_dyn, (void*)k_static
    };
    const char *names[] = {"simple", "coop_grid", "syncthreads", "cluster_2", "dyn_shmem", "static_shmem"};

    printf("# cudaFuncAttributes per kernel variant:\n");
    printf("# %-15s %-12s %-12s %-12s %-12s %-12s\n",
           "kernel", "shared", "max_dyn", "regs", "constSize", "ptxVersion");
    for (int i = 0; i < 6; i++) {
        cudaFuncGetAttributes(&fa, kernels[i]);
        printf("  %-15s %-12zu %-12d %-12d %-12zu %-12d\n",
               names[i], fa.sharedSizeBytes, fa.maxDynamicSharedSizeBytes,
               fa.numRegs, fa.constSizeBytes, fa.ptxVersion);
    }

    // ===== Test occupancy with various shmem requirements =====
    printf("\n# Occupancy probe: 256 threads/block, sweep dyn shmem size\n");
    cudaFuncSetAttribute((void*)k_dyn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)prop.sharedMemPerBlockOptin);

    int dyn_arr[] = {0, 1024, 4096, 16384, 32768, 49152, 65536, 100000, 116224};
    for (int dyn : dyn_arr) {
        int act_dyn, act_simple;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act_dyn, (void*)k_dyn, 256, dyn);
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act_simple, (void*)k_simple, 256, dyn);

        // Total shmem used per block
        size_t total = dyn + prop.reservedSharedMemPerBlock;
        // Max blocks based on shmem limit
        int sm_total = prop.sharedMemPerMultiprocessor;
        int max_by_shmem = sm_total / total;

        printf("  dyn=%-7d (per-block=%zu): k_dyn=%d, k_simple=%d (shmem-based max=%d)\n",
               dyn, total, act_dyn, act_simple, max_by_shmem);
    }

    // ===== Now test: cooperative kernel —is there extra shmem use? =====
    printf("\n# Compare occupancy: k_simple vs k_coop vs k_cluster (no dyn shmem)\n");
    int act_simple, act_coop, act_cluster, act_static;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act_simple, (void*)k_simple, 256, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act_coop, (void*)k_coop, 256, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act_cluster, (void*)k_cluster, 256, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&act_static, (void*)k_static, 256, 0);
    printf("  k_simple:  %d blocks/SM\n", act_simple);
    printf("  k_coop:    %d blocks/SM (uses this_grid)\n", act_coop);
    printf("  k_cluster: %d blocks/SM (uses cluster)\n", act_cluster);
    printf("  k_static:  %d blocks/SM (1024 bytes static)\n", act_static);

    // ===== Run the kernels to see what they actually use =====
    printf("\n# Run sample kernels\n");
    k_simple<<<1, 32>>>();
    cudaDeviceSynchronize();
    k_static<<<1, 32>>>();
    cudaDeviceSynchronize();

    return 0;
}
