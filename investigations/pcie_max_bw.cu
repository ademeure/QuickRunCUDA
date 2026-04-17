// Try to maximize PCIe H2D bandwidth
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    // Allocate large pinned buffer + device buffer
    size_t bytes = 1024 * 1024 * 1024;  // 1 GB
    void *h_pinned;
    cudaError_t err = cudaMallocHost(&h_pinned, bytes);
    if (err) { printf("MallocHost: %s\n", cudaGetErrorString(err)); return 1; }
    void *d;
    cudaMalloc(&d, bytes);

    // Touch host buffer
    memset(h_pinned, 0xAB, bytes);

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

    printf("# B300 PCIe H2D BW with various transfer sizes (best of 5)\n\n");
    printf("# %-12s %-12s %-15s\n", "size", "time_ms", "BW_GB/s");

    for (size_t mb : {16ul, 64ul, 256ul, 1024ul}) {
        size_t sz = mb * 1024 * 1024;
        if (sz > bytes) continue;

        float t = bench([&]{
            cudaMemcpyAsync(d, h_pinned, sz, cudaMemcpyHostToDevice, s);
        });
        printf("  %-12zu %-12.2f %-15.1f\n", mb, t, sz/(t/1000)/1e9);
    }

    // 3 stream pipelined
    printf("\n## 3 stream pipelined (each chunk 256 MB)\n");
    {
        cudaStream_t s1, s2, s3;
        cudaStreamCreate(&s1); cudaStreamCreate(&s2); cudaStreamCreate(&s3);
        size_t chunk = 256 * 1024 * 1024;

        float t = bench([&]{
            cudaMemcpyAsync((char*)d + 0*chunk, (char*)h_pinned + 0*chunk, chunk, cudaMemcpyHostToDevice, s1);
            cudaMemcpyAsync((char*)d + 1*chunk, (char*)h_pinned + 1*chunk, chunk, cudaMemcpyHostToDevice, s2);
            cudaMemcpyAsync((char*)d + 2*chunk, (char*)h_pinned + 2*chunk, chunk, cudaMemcpyHostToDevice, s3);
        });
        size_t total = 3 * chunk;
        printf("  3 × 256 MB H2D: %.2f ms = %.1f GB/s\n", t, total/(t/1000)/1e9);
    }

    return 0;
}
