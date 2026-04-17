// Various sync variants
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench_idle = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 5; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ns = std::chrono::duration<float, std::nano>(t1-t0).count();
            if (ns < best) best = ns;
        }
        return best;
    };

    auto bench_with_kernel = [&](auto sync_fn, int trials = 100) {
        for (int i = 0; i < 5; i++) {
            noop<<<1, 32, 0, s>>>();
            sync_fn();
        }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            noop<<<1, 32, 0, s>>>();
            sync_fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 sync variants — idle and after-kernel\n\n");
    printf("# %-30s %-15s %-15s\n", "method", "idle_ns", "after_kernel_us");

    printf("  %-30s %-15.0f ", "cudaDeviceSynchronize",
           bench_idle([&]{ cudaDeviceSynchronize(); }));
    printf("%.2f\n", bench_with_kernel([&]{ cudaDeviceSynchronize(); }));

    printf("  %-30s %-15.0f ", "cudaStreamSynchronize",
           bench_idle([&]{ cudaStreamSynchronize(s); }));
    printf("%.2f\n", bench_with_kernel([&]{ cudaStreamSynchronize(s); }));

    printf("  %-30s %-15.0f ", "cudaStreamSynchronize(NULL)",
           bench_idle([&]{ cudaStreamSynchronize(0); }));
    printf("%.2f\n", bench_with_kernel([&]{ cudaStreamSynchronize(0); }));

    cudaEvent_t e;
    cudaEventCreateWithFlags(&e, cudaEventDisableTiming);
    printf("  %-30s %-15.0f ", "cudaEventRecord+Sync (NoTime)",
           bench_idle([&]{ cudaEventRecord(e, s); cudaEventSynchronize(e); }));
    printf("%.2f\n", bench_with_kernel([&]{ cudaEventRecord(e, s); cudaEventSynchronize(e); }));

    return 0;
}
