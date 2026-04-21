// V8 H2: Host pinned vs unified memory transfer overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    size_t SIZE = 16 * 1024 * 1024;  // 16 MB

    // Test 1: pinned host → device
    void *pinned, *dev;
    cudaMallocHost(&pinned, SIZE);
    cudaMalloc(&dev, SIZE);
    memset(pinned, 0xAB, SIZE);

    cudaStream_t s; cudaStreamCreate(&s);
    int N = 100;

    // Warmup
    cudaMemcpyAsync(dev, pinned, SIZE, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) cudaMemcpyAsync(dev, pinned, SIZE, cudaMemcpyHostToDevice, s);
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double pinned_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // Test 2: pageable host → device
    void* pageable = malloc(SIZE);
    memset(pageable, 0xCD, SIZE);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) cudaMemcpy(dev, pageable, SIZE, cudaMemcpyHostToDevice);
    auto t3 = std::chrono::high_resolution_clock::now();
    double pageable_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    // Test 3: unified managed memory
    void* uma;
    cudaMallocManaged(&uma, SIZE);
    memset(uma, 0xEF, SIZE);
    cudaMemLocation loc = {cudaMemLocationTypeDevice, 0};
    cudaMemAdvise(uma, SIZE, cudaMemAdviseSetReadMostly, loc);
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaMemPrefetchAsync(uma, SIZE, loc, 0, s);
    }
    cudaStreamSynchronize(s);
    auto t5 = std::chrono::high_resolution_clock::now();
    double uma_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N;

    printf("16 MB host→device transfer:\n");
    printf("  Pinned (cudaMallocHost):     %7.1f us = %5.1f GB/s\n", pinned_us, SIZE / (pinned_us / 1e6) / 1e9);
    printf("  Pageable (malloc):           %7.1f us = %5.1f GB/s\n", pageable_us, SIZE / (pageable_us / 1e6) / 1e9);
    printf("  Unified (Managed+Prefetch):  %7.1f us = %5.1f GB/s\n", uma_us, SIZE / (uma_us / 1e6) / 1e9);

    return 0;
}
