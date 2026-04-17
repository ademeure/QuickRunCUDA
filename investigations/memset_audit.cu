// Audit: cudaMemset 7.47 TB/s really at HBM peak?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 5) {
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

    printf("# AUDIT: cudaMemset BW vs size (multi-trial, larger sizes)\n");
    printf("# HBM3E theoretical: 7672 GB/s\n\n");
    printf("# %-12s %-12s %-12s\n", "size_MB", "ms", "BW_GB/s");

    for (size_t mb : {64, 256, 1024, 4096, 16384}) {
        size_t bytes = (size_t)mb * 1024 * 1024;
        if (bytes > 100ull*1024*1024*1024) continue;
        unsigned char *d; cudaMalloc(&d, bytes);

        float t = bench([&]{
            cudaMemsetAsync(d, 0xAB, bytes, s);
        });
        printf("  %-12zu %-12.3f %-12.0f\n", mb, t, bytes/(t/1000)/1e9);
        cudaFree(d);
    }

    return 0;
}
