// cudaMallocAsync cross-stream behavior
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s1, s2;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

    // Set high release threshold so memory stays in pool
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);
    size_t big = 1024ul * 1024 * 1024;
    cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &big);

    auto bench = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 10; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        cudaDeviceSynchronize();
        return best;
    };

    printf("# B300 cudaMallocAsync cross-stream patterns\n\n");

    {
        float t = bench([&]{
            void *p; cudaMallocAsync(&p, 4096, s1);
            cudaFreeAsync(p, s1);
        });
        printf("  Alloc + Free same stream:        %.2f us\n", t);
    }
    {
        float t = bench([&]{
            void *p; cudaMallocAsync(&p, 4096, s1);
            cudaFreeAsync(p, s2);  // free on different stream
        });
        printf("  Alloc s1 + Free s2:              %.2f us\n", t);
    }
    {
        // Test pool reuse efficiency
        float t = bench([&]{
            void *p1, *p2, *p3, *p4;
            cudaMallocAsync(&p1, 4096, s1);
            cudaMallocAsync(&p2, 4096, s1);
            cudaMallocAsync(&p3, 4096, s1);
            cudaMallocAsync(&p4, 4096, s1);
            cudaFreeAsync(p4, s1);
            cudaFreeAsync(p3, s1);
            cudaFreeAsync(p2, s1);
            cudaFreeAsync(p1, s1);
        });
        printf("  4 alloc + 4 free same stream:    %.2f us = %.2f us each\n", t, t/8);
    }
    {
        // Test alternating sizes
        float t = bench([&]{
            void *p1, *p2;
            cudaMallocAsync(&p1, 4096, s1);
            cudaMallocAsync(&p2, 65536, s1);
            cudaFreeAsync(p2, s1);
            cudaFreeAsync(p1, s1);
        });
        printf("  Alt sizes (4K + 64K):            %.2f us\n", t);
    }

    return 0;
}
