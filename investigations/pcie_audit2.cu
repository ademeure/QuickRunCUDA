#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    void *h;
    cudaError_t err = cudaMallocHost(&h, 1ull * 1024 * 1024 * 1024);
    if (err) { printf("MallocHost failed: %s\n", cudaGetErrorString(err)); return 1; }
    void *d; cudaMalloc(&d, 1ull * 1024 * 1024 * 1024);

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

    printf("# PCIe Gen6 BW audit (1 GB pinned)\n\n");

    // Single transfer, sweep size
    printf("## Single H2D transfers:\n");
    for (size_t mb : {1, 16, 256, 1024}) {
        size_t bytes = mb * 1024 * 1024;
        float t = bench([&]{
            cudaMemcpyAsync(d, h, bytes, cudaMemcpyHostToDevice, s);
        });
        printf("  %zu MB: %.2f ms = %.1f GB/s\n", mb, t, bytes/(t/1000)/1e9);
    }

    // Multiple streams
    printf("\n## Multiple simultaneous streams (256 MB each):\n");
    cudaStream_t streams[8];
    for (int i = 0; i < 8; i++) cudaStreamCreate(&streams[i]);
    for (int n_streams : {1, 2, 4}) {
        size_t per_stream = 256 * 1024 * 1024;
        if (n_streams * per_stream > 1024ull * 1024 * 1024) continue;
        float t = bench([&]{
            for (int i = 0; i < n_streams; i++) {
                cudaMemcpyAsync((char*)d + i*per_stream, (char*)h + i*per_stream, per_stream, cudaMemcpyHostToDevice, streams[i]);
            }
        });
        size_t total = n_streams * per_stream;
        printf("  %d streams × 256 MB = %zu MB: %.2f ms = %.1f GB/s\n",
               n_streams, total/(1024*1024), t, total/(t/1000)/1e9);
    }

    return 0;
}
