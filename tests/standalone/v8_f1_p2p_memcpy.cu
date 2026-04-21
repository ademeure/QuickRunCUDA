// V8 F1: cudaMemcpyAsync peer-to-peer between 2 GPUs on NVLink
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    int n;
    cudaGetDeviceCount(&n);
    if (n < 2) { printf("Need 2 GPUs\n"); return 0; }

    // Enable peer access
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    // Alloc on each
    cudaSetDevice(0);
    float* dev0;
    cudaMalloc(&dev0, 1UL << 30);
    cudaSetDevice(1);
    float* dev1;
    cudaMalloc(&dev1, 1UL << 30);

    size_t sizes[] = {1024, 65536, 1048576, 16777216, 268435456, 1073741824};
    const char* labels[] = {"1 KB", "64 KB", "1 MB", "16 MB", "256 MB", "1 GB"};

    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    // Warmup
    cudaMemcpyPeerAsync(dev1, 1, dev0, 0, 4096, s);
    cudaStreamSynchronize(s);

    printf("GPU0 → GPU1 NVLink peer copy:\n");
    printf("%-10s %-12s %-12s\n", "Size", "Time (us)", "BW (GB/s)");

    for (int i = 0; i < 6; i++) {
        size_t sz = sizes[i];
        int N = (sz < 1048576) ? 1000 : (sz < 16777216 ? 100 : 20);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int r = 0; r < N; r++) {
            cudaMemcpyPeerAsync(dev1, 1, dev0, 0, sz, s);
        }
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        double bw = sz / (us / 1e6) / 1e9;
        printf("%-10s %-12.2f %-12.1f\n", labels[i], us, bw);
    }

    return 0;
}
