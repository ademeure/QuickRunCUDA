// CUDA IPC: handle creation costs
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    void *d_buf;
    cudaError_t err = cudaMalloc(&d_buf, 64 * 1024 * 1024);
    printf("# alloc: %s\n", cudaGetErrorString(err));

    cudaIpcMemHandle_t mh;
    err = cudaIpcGetMemHandle(&mh, d_buf);
    printf("# IpcGetMemHandle: %s\n", cudaGetErrorString(err));

    if (err != cudaSuccess) return 1;

    auto bench = [&](auto fn, int trials = 100) {
        for (int i = 0; i < 5; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    float t_get = bench([&]{ cudaIpcGetMemHandle(&mh, d_buf); });
    printf("  cudaIpcGetMemHandle:    %.2f us\n", t_get);

    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming | cudaEventInterprocess);
    cudaIpcEventHandle_t eh;
    float t_eget = bench([&]{ cudaIpcGetEventHandle(&eh, e); });
    printf("  cudaIpcGetEventHandle:  %.2f us\n", t_eget);

    return 0;
}
