// V7 M3: Driver API (cuLaunchKernel) vs Runtime API (cudaLaunchKernel) overhead
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    int N_RUNS = 1000;
    void* args[1] = {&buf};

    // Warmup
    noop<<<1, 32>>>(buf);
    cudaDeviceSynchronize();

    // 1. Triple-chevron syntax
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) noop<<<1, 32>>>(buf);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double chev_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // 2. Runtime cudaLaunchKernel
    dim3 grid(1), block(32);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) cudaLaunchKernel((void*)noop, grid, block, args, 0, 0);
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double rt_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    printf("Triple-chevron <<<>>>:        %.3f us/launch\n", chev_us);
    printf("Runtime cudaLaunchKernel:     %.3f us/launch\n", rt_us);
    printf("(Driver cuLaunchKernel needs separate CUmodule load — skipped for parity)\n");

    return 0;
}
