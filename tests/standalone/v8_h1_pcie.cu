// V8 H1: PCIe Gen 6 x16 transfer size sweep
// Measure host→device throughput at sizes 1 KB to 1 GB
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    size_t sizes[] = {1024UL, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216, 67108864, 268435456, 1073741824};
    const char* labels[] = {"1 KB", "4 KB", "16 KB", "64 KB", "256 KB", "1 MB", "4 MB", "16 MB", "64 MB", "256 MB", "1 GB"};

    void* host_pinned;
    void* dev;
    cudaMallocHost(&host_pinned, 1UL << 30);
    cudaMalloc(&dev, 1UL << 30);

    cudaStream_t s;
    cudaStreamCreate(&s);

    printf("PCIe Gen 6 x16 (B300) host→device transfer:\n");
    printf("%-10s %-15s %-15s\n", "Size", "Time (us)", "Throughput (GB/s)");

    for (int idx = 0; idx < 11; idx++) {
        size_t sz = sizes[idx];
        int N = (sz < 1048576) ? 1000 : (sz < 16777216 ? 100 : 10);

        // Warmup
        cudaMemcpyAsync(dev, host_pinned, sz, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            cudaMemcpyAsync(dev, host_pinned, sz, cudaMemcpyHostToDevice, s);
        }
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double per_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        double bw_gbps = sz / (per_us / 1e6) / 1e9;
        printf("%-10s %-15.2f %-15.1f\n", labels[idx], per_us, bw_gbps);
    }

    return 0;
}
