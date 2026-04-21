// V5 H3: Memory allocator perf — cudaMalloc vs cudaMallocAsync
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);

    int N = 10000;

    // Warmup
    void* d;
    cudaMalloc(&d, 4096);
    cudaFree(d);
    cudaMallocAsync(&d, 4096, s);
    cudaFreeAsync(d, s);
    cudaStreamSynchronize(s);

    // Test 1: cudaMalloc + cudaFree (size 4 KB)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaMalloc(&d, 4096);
        cudaFree(d);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sync_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    printf("cudaMalloc + cudaFree (4 KB): %.2f us each\n", sync_us);

    // Test 2: cudaMallocAsync + cudaFreeAsync
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaMallocAsync(&d, 4096, s);
        cudaFreeAsync(d, s);
    }
    cudaStreamSynchronize(s);
    auto t3 = std::chrono::high_resolution_clock::now();
    double async_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;
    printf("cudaMallocAsync + cudaFreeAsync (4 KB): %.2f us each\n", async_us);

    // Test 3: cudaMalloc with 1 MB allocations
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        cudaMalloc(&d, 1024 * 1024);
        cudaFree(d);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double sync_1m = std::chrono::duration<double, std::micro>(t5 - t4).count() / 1000;
    printf("cudaMalloc + cudaFree (1 MB): %.2f us each\n", sync_1m);

    // Test 4: cudaMallocAsync with 1 MB
    auto t6 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 1000; i++) {
        cudaMallocAsync(&d, 1024 * 1024, s);
        cudaFreeAsync(d, s);
    }
    cudaStreamSynchronize(s);
    auto t7 = std::chrono::high_resolution_clock::now();
    double async_1m = std::chrono::duration<double, std::micro>(t7 - t6).count() / 1000;
    printf("cudaMallocAsync + cudaFreeAsync (1 MB): %.2f us each\n", async_1m);

    return 0;
}
