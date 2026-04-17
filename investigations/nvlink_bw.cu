// NVLink P2P bandwidth between 2 B300 GPUs
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    // Allocate buffers on both GPUs
    cudaSetDevice(0);
    void *d0; cudaMalloc(&d0, 1024ull * 1024 * 1024);  // 1 GB on GPU 0
    cudaSetDevice(1);
    void *d1; cudaMalloc(&d1, 1024ull * 1024 * 1024);  // 1 GB on GPU 1

    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 10) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 NVLink P2P bandwidth (GPU0 ↔ GPU1)\n");
    printf("# Topology: NV18 (multi-link NVLink between B300s)\n\n");

    // Read from GPU 1 to GPU 0
    for (size_t mb : {16, 256, 1024}) {
        size_t bytes = mb * 1024 * 1024;
        float t = bench([&]{
            cudaMemcpyPeerAsync(d0, 0, d1, 1, bytes, s);
        });
        printf("  %zu MB GPU1→GPU0: %.2f ms = %.0f GB/s\n", mb, t, bytes/(t/1000)/1e9);
    }

    // Other direction
    for (size_t mb : {1024}) {
        size_t bytes = mb * 1024 * 1024;
        float t = bench([&]{
            cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s);
        });
        printf("  %zu MB GPU0→GPU1: %.2f ms = %.0f GB/s\n", mb, t, bytes/(t/1000)/1e9);
    }

    // Bidirectional
    {
        cudaStream_t s1; cudaStreamCreate(&s1);
        size_t bytes = 1024ull * 1024 * 1024;
        float t = bench([&]{
            cudaMemcpyPeerAsync(d0, 0, d1, 1, bytes, s);
            cudaMemcpyPeerAsync(d1, 1, d0, 0, bytes, s1);
        });
        printf("  1024 MB BIDIR: %.2f ms = %.0f GB/s aggregate\n", t, 2.0*bytes/(t/1000)/1e9);
    }

    return 0;
}
