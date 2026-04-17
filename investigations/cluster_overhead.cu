// Cluster launch overhead vs regular launch
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop_regular() {}
extern "C" __global__ void __cluster_dims__(2,1,1) noop_cluster_2() {}
extern "C" __global__ void __cluster_dims__(4,1,1) noop_cluster_4() {}
extern "C" __global__ void __cluster_dims__(8,1,1) noop_cluster_8() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 200) {
        for (int i = 0; i < 5; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    auto bench_async = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 5; i++) fn();
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < trials; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();
        cudaStreamSynchronize(s);
        return us / trials;
    };

    printf("# B300 cluster launch overhead\n\n");
    printf("# %-30s %-12s %-12s\n", "method", "sync_us", "async_us");

    printf("  %-30s %.2f         %.2f\n", "regular noop (1 block)",
           bench([&]{ noop_regular<<<1, 32, 0, s>>>(); }),
           bench_async([&]{ noop_regular<<<1, 32, 0, s>>>(); }));
    printf("  %-30s %.2f         %.2f\n", "cluster_2 noop (2 blocks)",
           bench([&]{ noop_cluster_2<<<2, 32, 0, s>>>(); }),
           bench_async([&]{ noop_cluster_2<<<2, 32, 0, s>>>(); }));
    printf("  %-30s %.2f         %.2f\n", "cluster_4 noop (4 blocks)",
           bench([&]{ noop_cluster_4<<<4, 32, 0, s>>>(); }),
           bench_async([&]{ noop_cluster_4<<<4, 32, 0, s>>>(); }));
    printf("  %-30s %.2f         %.2f\n", "cluster_8 noop (8 blocks)",
           bench([&]{ noop_cluster_8<<<8, 32, 0, s>>>(); }),
           bench_async([&]{ noop_cluster_8<<<8, 32, 0, s>>>(); }));

    // Compare to regular with same block count
    printf("\n## Regular launch with same block count\n");
    printf("  %-30s %.2f         %.2f\n", "regular 2 blocks",
           bench([&]{ noop_regular<<<2, 32, 0, s>>>(); }),
           bench_async([&]{ noop_regular<<<2, 32, 0, s>>>(); }));
    printf("  %-30s %.2f         %.2f\n", "regular 4 blocks",
           bench([&]{ noop_regular<<<4, 32, 0, s>>>(); }),
           bench_async([&]{ noop_regular<<<4, 32, 0, s>>>(); }));
    printf("  %-30s %.2f         %.2f\n", "regular 8 blocks",
           bench([&]{ noop_regular<<<8, 32, 0, s>>>(); }),
           bench_async([&]{ noop_regular<<<8, 32, 0, s>>>(); }));

    return 0;
}
