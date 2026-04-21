// V8 F4: IPC handle creation/open at scale
// Measure if creating many handles scales linearly or hits driver contention
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    int N_BUFS = 100;
    void* bufs[100];
    cudaIpcMemHandle_t handles[100];

    // Allocate N buffers
    for (int i = 0; i < N_BUFS; i++) {
        cudaMalloc(&bufs[i], 1024 * 1024);  // 1 MB each
    }

    // Time IpcGetMemHandle × N
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_BUFS; i++) {
        cudaIpcGetMemHandle(&handles[i], bufs[i]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double get_total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    printf("IPC handle creation × %d: total %.1f us, per-handle %.2f us\n",
           N_BUFS, get_total_us, get_total_us / N_BUFS);

    // Compare to single handle (first call cost vs subsequent)
    void* extra;
    cudaMalloc(&extra, 1024 * 1024);
    cudaIpcMemHandle_t h1, h2;
    auto t2 = std::chrono::high_resolution_clock::now();
    cudaIpcGetMemHandle(&h1, extra);
    auto t3 = std::chrono::high_resolution_clock::now();
    cudaIpcGetMemHandle(&h2, extra);
    auto t4 = std::chrono::high_resolution_clock::now();
    printf("Single handle first call: %.2f us\n",
           std::chrono::duration<double, std::micro>(t3 - t2).count());
    printf("Single handle second call (warm): %.2f us\n",
           std::chrono::duration<double, std::micro>(t4 - t3).count());

    return 0;
}
