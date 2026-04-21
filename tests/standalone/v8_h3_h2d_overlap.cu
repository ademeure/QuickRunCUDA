// V8 H3: Async H2D + kernel overlap patterns
// Compare: serial (H2D, sync, kernel) vs overlapped (H2D + kernel in different streams)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void compute(float* buf, int n, int delay_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do { asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)delay_iters);
        if (n > 0) buf[0] = 1.0f;
    }
}

int main() {
    cudaSetDevice(0);

    size_t SIZE = 16 * 1024 * 1024;  // 16 MB transfer (~ 280 µs)
    float *host, *dev;
    cudaMallocHost(&host, SIZE);
    cudaMalloc(&dev, SIZE);

    int N = 50;
    int delay = 1500000;  // ~1 ms compute

    // Test 1: serial (H2D + kernel in same stream)
    cudaStream_t s1; cudaStreamCreate(&s1);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaMemcpyAsync(dev, host, SIZE, cudaMemcpyHostToDevice, s1);
        compute<<<1, 32, 0, s1>>>(dev, SIZE / 4, delay);
    }
    cudaStreamSynchronize(s1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double serial_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // Test 2: 2 streams (H2D in s1, compute in s2 with cross-stream wait)
    cudaStream_t s2; cudaStreamCreate(&s2);
    cudaEvent_t e; cudaEventCreate(&e);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaMemcpyAsync(dev, host, SIZE, cudaMemcpyHostToDevice, s1);
        cudaEventRecord(e, s1);
        cudaStreamWaitEvent(s2, e, 0);
        compute<<<1, 32, 0, s2>>>(dev, SIZE / 4, delay);
    }
    cudaStreamSynchronize(s2);
    auto t3 = std::chrono::high_resolution_clock::now();
    double overlap_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();

    printf("16 MB H2D + 1 ms compute, %d iters:\n", N);
    printf("  Serial (same stream):    %.2f ms total = %.2f ms/iter\n", serial_ms, serial_ms / N);
    printf("  2 streams + event:       %.2f ms total = %.2f ms/iter\n", overlap_ms, overlap_ms / N);
    printf("  Speedup: %.2fx\n", serial_ms / overlap_ms);

    return 0;
}
