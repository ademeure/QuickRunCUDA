// Effect of CUDA_LAUNCH_BLOCKING=1
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstdlib>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    const char *env_val = getenv("CUDA_LAUNCH_BLOCKING");
    printf("# CUDA_LAUNCH_BLOCKING: %s\n", env_val ? env_val : "(unset)");

    auto bench = [&](auto fn, int trials = 200) {
        for (int i = 0; i < 5; i++) fn();
        cudaDeviceSynchronize();
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

    // Pure async submission
    {
        // Warm up
        for (int i = 0; i < 100; i++) noop<<<1, 32, 0, s>>>();
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; i++) noop<<<1, 32, 0, s>>>();
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();
        printf("  1000 launches submission: %.0f us = %.2f us/launch\n", us, us/1000);

        cudaStreamSynchronize(s);
    }

    // Single launch + sync
    float t = bench([&]{
        noop<<<1, 32, 0, s>>>();
        cudaStreamSynchronize(s);
    });
    printf("  Single launch + sync: %.2f us\n", t);

    return 0;
}
