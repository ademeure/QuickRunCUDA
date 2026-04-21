// V7 M5: Memory pool allocator detailed perf
// Test cudaMallocAsync vs cudaMallocFromPoolAsync vs custom pool
// Various sizes 4 KB to 256 MB
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);

    // Get default mempool
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);

    int N_RUNS = 1000;
    size_t sizes[] = {4096, 65536, 1048576, 16777216, 268435456};
    const char* size_labels[] = {"4 KB", "64 KB", "1 MB", "16 MB", "256 MB"};

    printf("Memory allocator performance:\n");
    printf("%-10s %-15s %-15s %-15s\n", "Size", "MallocAsync(us)", "FromPool(us)", "MallocSync(us)");

    for (int idx = 0; idx < 5; idx++) {
        size_t sz = sizes[idx];
        void* ptrs[1000];

        // 1. cudaMallocAsync (default pool)
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_RUNS; i++) {
            cudaMallocAsync(&ptrs[i], sz, s);
        }
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double malloc_async_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

        for (int i = 0; i < N_RUNS; i++) cudaFreeAsync(ptrs[i], s);
        cudaStreamSynchronize(s);

        // 2. cudaMallocFromPoolAsync (explicit pool)
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_RUNS; i++) {
            cudaMallocFromPoolAsync(&ptrs[i], sz, pool, s);
        }
        cudaStreamSynchronize(s);
        auto t3 = std::chrono::high_resolution_clock::now();
        double from_pool_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

        for (int i = 0; i < N_RUNS; i++) cudaFreeAsync(ptrs[i], s);
        cudaStreamSynchronize(s);

        // 3. cudaMalloc (sync, 100 runs only — slow)
        int N_SYNC = 100;
        auto t4 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N_SYNC; i++) {
            cudaMalloc(&ptrs[i], sz);
        }
        auto t5 = std::chrono::high_resolution_clock::now();
        double malloc_sync_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N_SYNC;

        for (int i = 0; i < N_SYNC; i++) cudaFree(ptrs[i]);

        printf("%-10s %-15.3f %-15.3f %-15.3f\n",
               size_labels[idx], malloc_async_us, from_pool_us, malloc_sync_us);
    }

    return 0;
}
