#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void manual_zero(unsigned long long *p, size_t N) {
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = tid; i < N; i += stride) p[i] = 0;
}

extern "C" __global__ void manual_zero_v4(unsigned long long *p, size_t N) {
    // Use vectorized ulonglong2 stores
    ulonglong2 *p2 = (ulonglong2*)p;
    size_t N2 = N / 2;
    size_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    ulonglong2 zero = {0, 0};
    for (size_t i = tid; i < N2; i += stride) p2[i] = zero;
}

int main() {
    cudaSetDevice(0);

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 2; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    cudaStream_t s; cudaStreamCreate(&s);

    printf("# B300 memset variants (write-only bandwidth)\n");
    printf("# %-15s %-15s %-15s %-15s\n", "size", "memset_GB/s", "manual_GB/s", "manualV4_GB/s");

    for (size_t mb : {1, 16, 256, 1024, 4096}) {
        size_t bytes = (size_t)mb * 1024 * 1024;
        unsigned long long *d; cudaMalloc(&d, bytes);

        float t_set = bench([&]{
            cudaMemsetAsync(d, 0, bytes, s);
        });
        float t_man = bench([&]{
            manual_zero<<<148, 256, 0, s>>>(d, bytes / 8);
        });
        float t_v4 = bench([&]{
            manual_zero_v4<<<148, 256, 0, s>>>(d, bytes / 8);
        });

        printf("  %-15zu %-15.0f %-15.0f %-15.0f\n",
               mb, bytes/(t_set/1000)/1e9, bytes/(t_man/1000)/1e9, bytes/(t_v4/1000)/1e9);

        cudaFree(d);
    }

    return 0;
}
