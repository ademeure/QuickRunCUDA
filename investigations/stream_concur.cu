// Compare: 100 kernels back-to-back vs same kernel × 100
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

extern "C" __global__ void busy(unsigned long long *out, int cycles) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long *d_out; cudaMalloc(&d_out, 1024*sizeof(unsigned long long));

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 10) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    int delay = 100 * 2032;  // 100 us each kernel

    printf("# B300 sequential vs concurrent kernel chains (each ~100us, single block)\n\n");
    printf("# %-15s %-15s %-15s %-15s\n", "n_kernels", "method", "wall_us", "per_kernel_us");

    // Sequential same stream
    for (int N : {10, 100, 1000}) {
        float t = bench([&]{
            for (int i = 0; i < N; i++) busy<<<1, 32, 0, s>>>(d_out, delay);
        });
        printf("  %-15d %-15s %-15.0f %-15.2f\n", N, "1 stream", t, t/N);
    }

    // Concurrent across many streams (up to 128 limit)
    std::vector<cudaStream_t> ss(128);
    for (int i = 0; i < 128; i++) cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking);

    for (int N : {16, 64, 128}) {
        float t = bench([&]{
            for (int i = 0; i < N; i++) busy<<<1, 32, 0, ss[i]>>>(d_out, delay);
        });
        printf("  %-15d %-15s %-15.0f %-15.2f\n", N, "N streams", t, t/N);
    }

    return 0;
}
