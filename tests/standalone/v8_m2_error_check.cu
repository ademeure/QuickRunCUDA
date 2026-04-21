// V8 M2: Async error checking — cudaPeekAtLastError + GetLastError perf
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    int N = 10000;

    // Warmup
    noop<<<1, 32>>>(buf);
    cudaDeviceSynchronize();

    // 1. cudaPeekAtLastError (async, no sync)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        noop<<<1, 32>>>(buf);
        cudaPeekAtLastError();
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double peek_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // 2. cudaGetLastError (resets error state)
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        noop<<<1, 32>>>(buf);
        cudaGetLastError();
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double get_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    // 3. No error check
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        noop<<<1, 32>>>(buf);
    }
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double none_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N;

    // 4. Per-launch synchronize + error check
    auto t6 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N / 10; i++) {
        noop<<<1, 32>>>(buf);
        cudaDeviceSynchronize();
        cudaGetLastError();
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    double sync_check_us = std::chrono::duration<double, std::micro>(t7 - t6).count() / (N / 10);

    printf("Per-launch overhead:\n");
    printf("  No error check:           %.3f us\n", none_us);
    printf("  cudaPeekAtLastError:      %.3f us\n", peek_us);
    printf("  cudaGetLastError:         %.3f us\n", get_us);
    printf("  Sync + GetLastError:      %.3f us\n", sync_check_us);

    return 0;
}
