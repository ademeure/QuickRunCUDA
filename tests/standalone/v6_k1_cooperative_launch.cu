// V6 K1: cudaLaunchCooperativeKernel overhead
// Compare:
//   - Direct launch (kernel<<<...>>>)
//   - cudaLaunchKernel (explicit)
//   - cudaLaunchCooperativeKernel (enables grid sync)
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>
#include <chrono>

namespace cg = cooperative_groups;

__global__ void noop_kernel(unsigned int* buf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[0] = 1;
}

__global__ void coop_kernel(unsigned int* buf) {
    auto grid = cg::this_grid();
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[0] = 1;
    grid.sync();  // Grid-wide barrier — only valid in cooperative launch
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 1024);

    int N_RUNS = 1000;
    int blocks = 148;
    int threads = 128;

    // Warmup
    noop_kernel<<<blocks, threads>>>(buf);
    cudaDeviceSynchronize();

    // 1. Direct launch
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        noop_kernel<<<blocks, threads>>>(buf);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double direct_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // 2. cudaLaunchKernel (explicit, non-cooperative)
    void* args[1] = {&buf};
    dim3 grid(blocks), block(threads);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        cudaLaunchKernel((void*)noop_kernel, grid, block, args, 0, 0);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double explicit_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    // 3. cudaLaunchCooperativeKernel
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        cudaLaunchCooperativeKernel((void*)coop_kernel, grid, block, args, 0, 0);
    }
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double coop_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N_RUNS;

    printf("Direct launch:                %.2f us/launch\n", direct_us);
    printf("cudaLaunchKernel (explicit):  %.2f us/launch\n", explicit_us);
    printf("cudaLaunchCooperativeKernel:  %.2f us/launch\n", coop_us);
    printf("Cooperative overhead vs direct: %.2fx\n", coop_us / direct_us);

    return 0;
}
