// L3: Driver-side per-kernel-launch overhead
// Compare: empty kernel vs minimal kernel; cuLaunchKernel vs cudaLaunchKernel; varying args
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

__global__ void empty_kernel() {}

__global__ void empty_kernel_with_args(int a, int b, int c) {
    if (a + b + c == 12345 && threadIdx.x == 0) {
        // never true at runtime
        printf("never\n");
    }
}

int main() {
    // Warmup
    empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    int N = 10000;

    // Test 1: cudaLaunch via <<<>>>, no args
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        empty_kernel<<<1, 1>>>();
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns_per = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
    printf("cudaLaunch (no args, sync at end): %.2f ns/launch\n", ns_per);

    // Test 2: with sync after each
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N/10; i++) {
        empty_kernel<<<1, 1>>>();
        cudaDeviceSynchronize();
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double ns_per_sync = std::chrono::duration<double, std::nano>(t3 - t2).count() / (N/10);
    printf("cudaLaunch + sync each: %.2f ns/launch\n", ns_per_sync);

    // Test 3: with 3 int args (vs no args)
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        empty_kernel_with_args<<<1, 1>>>(i, i+1, i+2);
    }
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double ns_per_args = std::chrono::duration<double, std::nano>(t5 - t4).count() / N;
    printf("cudaLaunch + 3 int args: %.2f ns/launch\n", ns_per_args);

    // Test 4: cuLaunchKernel (driver API)
    CUmodule mod;
    CUfunction func;
    cuModuleLoad(&mod, "/dev/null"); // dummy; will fail but that's OK
    // Skip driver API for now — function lookup adds complexity
    // Just measure runtime overhead

    return 0;
}
