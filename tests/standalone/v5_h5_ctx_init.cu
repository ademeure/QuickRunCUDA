// V5 H5: Driver context init — cold-start latency measurement
// Time from process start to first kernel completion
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop() {}

int main() {
    auto t0 = std::chrono::high_resolution_clock::now();

    cudaSetDevice(0);
    auto t1 = std::chrono::high_resolution_clock::now();

    // First kernel launch (will trigger context init if not already)
    noop<<<1, 1>>>();
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    // Second kernel launch (warm)
    noop<<<1, 1>>>();
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();

    double t_set_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    double t_first_us = std::chrono::duration<double, std::micro>(t2 - t1).count();
    double t_warm_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    printf("cudaSetDevice(0): %.1f us\n", t_set_us);
    printf("First kernel launch + sync: %.1f us\n", t_first_us);
    printf("Warm kernel launch + sync: %.1f us\n", t_warm_us);
    printf("Process start → first kernel done: %.1f us\n", t_set_us + t_first_us);
    return 0;
}
