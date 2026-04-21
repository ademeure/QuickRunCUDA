// V5 H4: cudaMallocFromPoolAsync overhead at various sizes
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);

    // Get default pool
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);

    int N = 10000;
    void* d;

    // Warmup
    for (int i = 0; i < 100; i++) {
        cudaMallocFromPoolAsync(&d, 4096, pool, s);
        cudaFreeAsync(d, s);
    }
    cudaStreamSynchronize(s);

    // Sweep sizes
    for (size_t sz : {(size_t)64, (size_t)4096, (size_t)65536, (size_t)1048576, (size_t)16777216}) {
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            cudaMallocFromPoolAsync(&d, sz, pool, s);
            cudaFreeAsync(d, s);
        }
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        printf("MallocFromPoolAsync + FreeAsync (size %8zu B): %.3f us each\n", sz, us);
    }

    return 0;
}
