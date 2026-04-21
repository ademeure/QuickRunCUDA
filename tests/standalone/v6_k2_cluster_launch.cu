// V6 K2: Cluster launch overhead (cudaLaunchKernelEx with cluster dim)
// Compare: direct launch, cudaLaunchKernelEx (no cluster), cudaLaunchKernelEx (cluster=2/4/8)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop_kernel(unsigned int* buf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 1024);

    int N_RUNS = 1000;
    int blocks = 16;  // need to be multiple of cluster size
    int threads = 128;

    // Warmup
    noop_kernel<<<blocks, threads>>>(buf);
    cudaDeviceSynchronize();

    void* args[1] = {&buf};
    dim3 grid(blocks), block(threads);

    // 1. Direct launch
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        noop_kernel<<<blocks, threads>>>(buf);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double direct_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // 2. cudaLaunchKernelEx without cluster
    cudaLaunchConfig_t config = {};
    config.gridDim = grid;
    config.blockDim = block;
    config.dynamicSmemBytes = 0;
    config.stream = 0;
    config.numAttrs = 0;

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        cudaLaunchKernelEx(&config, noop_kernel, buf);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double ex_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    // 3-5. With cluster size 2/4/8
    cudaLaunchAttribute attrs[1];
    attrs[0].id = cudaLaunchAttributeClusterDimension;
    config.numAttrs = 1;
    config.attrs = attrs;

    int sizes[] = {2, 4, 8};
    double cluster_us[3];

    for (int idx = 0; idx < 3; idx++) {
        attrs[0].val.clusterDim.x = sizes[idx];
        attrs[0].val.clusterDim.y = 1;
        attrs[0].val.clusterDim.z = 1;

        auto ts = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_RUNS; i++) {
            cudaLaunchKernelEx(&config, noop_kernel, buf);
        }
        cudaDeviceSynchronize();
        auto te = std::chrono::high_resolution_clock::now();
        cluster_us[idx] = std::chrono::duration<double, std::micro>(te - ts).count() / N_RUNS;
    }

    printf("Direct launch:                  %.2f us/launch\n", direct_us);
    printf("cudaLaunchKernelEx (no cluster):%.2f us/launch (%.2fx)\n", ex_us, ex_us/direct_us);
    printf("cudaLaunchKernelEx cluster=2:   %.2f us/launch (%.2fx)\n", cluster_us[0], cluster_us[0]/direct_us);
    printf("cudaLaunchKernelEx cluster=4:   %.2f us/launch (%.2fx)\n", cluster_us[1], cluster_us[1]/direct_us);
    printf("cudaLaunchKernelEx cluster=8:   %.2f us/launch (%.2fx)\n", cluster_us[2], cluster_us[2]/direct_us);

    return 0;
}
