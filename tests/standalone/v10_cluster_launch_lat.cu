// V10: cluster launch latency vs regular launch
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

__global__ void empty() {}
__global__ __cluster_dims__(4, 1, 1) void empty_cluster() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);
    int N = 1000;

    // Direct launch
    for (int i = 0; i < 10; i++) empty<<<8, 32, 0, s>>>();
    cudaStreamSynchronize(s);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        empty<<<8, 32, 0, s>>>();
        cudaStreamSynchronize(s);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double direct_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // Cluster launch via cudaLaunchKernelEx
    cudaLaunchConfig_t cfg = {};
    cfg.gridDim = dim3(8, 1, 1);
    cfg.blockDim = dim3(32, 1, 1);
    cfg.stream = s;
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    attrs[0].val.clusterDim.x = 4;
    attrs[0].val.clusterDim.y = 1;
    attrs[0].val.clusterDim.z = 1;
    cfg.attrs = attrs;
    cfg.numAttrs = 1;

    for (int i = 0; i < 10; i++) cudaLaunchKernelEx(&cfg, empty_cluster);
    cudaStreamSynchronize(s);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaLaunchKernelEx(&cfg, empty_cluster);
        cudaStreamSynchronize(s);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double cluster_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    printf("=== Launch latency (empty kernel + sync) ===\n");
    printf("  Direct empty<<<8,32>>>:    %.3f us\n", direct_us);
    printf("  Cluster (cluster=4):       %.3f us\n", cluster_us);
    printf("  Cluster overhead vs direct: %+.3f us (%.2fx)\n",
           cluster_us - direct_us, cluster_us / direct_us);

    cudaStreamDestroy(s);
    return 0;
}
