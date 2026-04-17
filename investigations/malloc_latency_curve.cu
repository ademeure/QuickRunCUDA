// cudaMalloc latency curve - small to huge
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

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

    printf("# B300 cudaMalloc / cudaMallocAsync latency vs size\n\n");
    printf("# %-15s %-15s %-15s %-12s\n",
           "size", "Malloc_us", "MallocAsync_us", "ratio");

    size_t sizes[] = {16ul, 256ul, 4096ul, 65536ul, 1024ul*1024,
                      16ul*1024*1024, 256ul*1024*1024, 1024ul*1024*1024,
                      4ul*1024*1024*1024, 16ul*1024*1024*1024};
    const char *labels[] = {"16 B", "256 B", "4 KB", "64 KB", "1 MB",
                            "16 MB", "256 MB", "1 GB", "4 GB", "16 GB"};

    for (int i = 0; i < 10; i++) {
        size_t sz = sizes[i];

        // Sync malloc/free
        float t_sync = bench([&]{
            void *p; cudaMalloc(&p, sz); cudaFree(p);
        }, 10);

        // Async malloc/free
        float t_async = bench([&]{
            void *p; cudaMallocAsync(&p, sz, s); cudaFreeAsync(p, s);
        }, 30);

        printf("  %-15s %-15.0f %-15.0f %.0fx\n",
               labels[i], t_sync, t_async, t_sync / t_async);
    }

    return 0;
}
