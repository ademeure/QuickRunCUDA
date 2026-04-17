// Cluster sync cost vs cluster size
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <chrono>

namespace cg = cooperative_groups;

extern "C" __global__ void __cluster_dims__(2,1,1) cluster_sync_2(uint64_t *out, int iters) {
    auto cluster = cg::this_cluster();
    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) {
        cluster.sync();
    }
    uint64_t t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0)
        out[blockIdx.x] = t1 - t0;
}

extern "C" __global__ void __cluster_dims__(4,1,1) cluster_sync_4(uint64_t *out, int iters) {
    auto cluster = cg::this_cluster();
    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) cluster.sync();
    uint64_t t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0)
        out[blockIdx.x] = t1 - t0;
}

extern "C" __global__ void __cluster_dims__(8,1,1) cluster_sync_8(uint64_t *out, int iters) {
    auto cluster = cg::this_cluster();
    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) cluster.sync();
    uint64_t t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0)
        out[blockIdx.x] = t1 - t0;
}

extern "C" __global__ void __cluster_dims__(16,1,1) cluster_sync_16(uint64_t *out, int iters) {
    auto cluster = cg::this_cluster();
    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) cluster.sync();
    uint64_t t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0)
        out[blockIdx.x] = t1 - t0;
}

extern "C" __global__ void block_sync(uint64_t *out, int iters) {
    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) {
        __syncthreads();
    }
    uint64_t t1 = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0)
        out[blockIdx.x] = t1 - t0;
}

int main() {
    cudaSetDevice(0);

    int max_cluster_size;
    cudaDeviceGetAttribute(&max_cluster_size, cudaDevAttrClusterLaunch, 0);
    printf("# B300 cluster launch supported: %d\n", max_cluster_size);

    // Skip clusterSize query - has tricky API requirements

    uint64_t *d_out; cudaMalloc(&d_out, 1024*sizeof(uint64_t));
    int iters = 1000;

    printf("\n# Sync cost per operation (1000 iter avg, 256 threads/block)\n");
    printf("# %-15s %-12s %-12s\n", "primitive", "cycles", "ns_at_2032");

    // Block sync
    {
        block_sync<<<1, 256>>>(d_out, iters);
        cudaDeviceSynchronize();
        uint64_t cyc; cudaMemcpy(&cyc, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-15s %-12.1f %-12.1f\n", "__syncthreads", per, per/2.032);
    }

    // Cluster syncs
    cluster_sync_2<<<2, 256>>>(d_out, iters);
    cudaDeviceSynchronize();
    {
        uint64_t cyc; cudaMemcpy(&cyc, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-15s %-12.1f %-12.1f\n", "cluster.sync 2", per, per/2.032);
    }
    cluster_sync_4<<<4, 256>>>(d_out, iters);
    cudaDeviceSynchronize();
    {
        uint64_t cyc; cudaMemcpy(&cyc, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-15s %-12.1f %-12.1f\n", "cluster.sync 4", per, per/2.032);
    }
    cluster_sync_8<<<8, 256>>>(d_out, iters);
    cudaDeviceSynchronize();
    {
        uint64_t cyc; cudaMemcpy(&cyc, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-15s %-12.1f %-12.1f\n", "cluster.sync 8", per, per/2.032);
    }
    cluster_sync_16<<<16, 256>>>(d_out, iters);
    cudaError_t err = cudaDeviceSynchronize();
    if (err == cudaSuccess) {
        uint64_t cyc; cudaMemcpy(&cyc, d_out, sizeof(uint64_t), cudaMemcpyDeviceToHost);
        double per = (double)cyc / iters;
        printf("  %-15s %-12.1f %-12.1f (non-portable max)\n", "cluster.sync 16", per, per/2.032);
    } else {
        printf("  cluster.sync 16: %s\n", cudaGetErrorString(err));
    }

    return 0;
}
