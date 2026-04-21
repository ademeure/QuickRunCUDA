// V8 H5: H2D + D2H in same stream vs different streams
// PCIe Gen 6 is bidirectional; should overlap H2D and D2H if separate streams
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    size_t SIZE = 16 * 1024 * 1024;  // 16 MB

    void *h2d_host, *d2h_host;
    void *dev_in, *dev_out;
    cudaMallocHost(&h2d_host, SIZE);
    cudaMallocHost(&d2h_host, SIZE);
    cudaMalloc(&dev_in, SIZE);
    cudaMalloc(&dev_out, SIZE);

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    int N = 50;

    // Warmup
    cudaMemcpyAsync(dev_in, h2d_host, SIZE, cudaMemcpyHostToDevice, s1);
    cudaStreamSynchronize(s1);

    // Test 1: same stream H2D then D2H (serial)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaMemcpyAsync(dev_in, h2d_host, SIZE, cudaMemcpyHostToDevice, s1);
        cudaMemcpyAsync(d2h_host, dev_out, SIZE, cudaMemcpyDeviceToHost, s1);
    }
    cudaStreamSynchronize(s1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Test 2: different streams (parallel)
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaMemcpyAsync(dev_in, h2d_host, SIZE, cudaMemcpyHostToDevice, s1);
        cudaMemcpyAsync(d2h_host, dev_out, SIZE, cudaMemcpyDeviceToHost, s2);
    }
    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    auto t3 = std::chrono::high_resolution_clock::now();
    double parallel_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    printf("16 MB H2D + D2H × %d:\n", N);
    printf("  Same stream (serial):    %.2f ms = %.2f ms/pair\n", serial_ms, serial_ms / N);
    printf("  Different streams:       %.2f ms = %.2f ms/pair\n", parallel_ms, parallel_ms / N);
    printf("  Overlap speedup: %.2fx\n", serial_ms / parallel_ms);

    return 0;
}
