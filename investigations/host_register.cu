// Compare cudaMallocHost vs cudaHostRegister for pinned memory
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstdlib>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto measure = [&](auto fn, int trials=20) {
        for (int i = 0; i < 3; i++) fn();
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < trials; i++) fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<float, std::micro>(t1-t0).count() / trials;
    };

    printf("# B300: cudaMallocHost vs cudaHostRegister\n\n");

    // Various sizes
    for (size_t sz : {(size_t)4096, (size_t)65536, (size_t)(1<<20), (size_t)(16<<20)}) {
        // cudaMallocHost
        float t_mallochost = measure([&]{
            void *p; cudaMallocHost(&p, sz);
            cudaFreeHost(p);
        });

        // cudaHostRegister on system-allocated buffer
        float t_register = measure([&]{
            void *p = malloc(sz);
            memset(p, 0, sz);  // ensure touched so not lazy-allocated
            cudaHostRegister(p, sz, cudaHostRegisterDefault);
            cudaHostUnregister(p);
            free(p);
        }, 5);

        printf("  size=%zu bytes:\n", sz);
        printf("    cudaMallocHost+Free    : %8.2f us\n", t_mallochost);
        printf("    cudaHostRegister+Unreg : %8.2f us\n", t_register);
        printf("    (malloc+memset not counted in register)\n\n");
    }

    // Test memcpy from both
    printf("## H2D memcpy throughput from each pinned-memory flavor (16 MB)\n");
    size_t sz = 16 * 1024 * 1024;
    void *d;
    cudaMalloc(&d, sz);

    // MallocHost
    void *mh;
    cudaMallocHost(&mh, sz);
    memset(mh, 0, sz);
    {
        for (int i = 0; i < 3; i++) cudaMemcpy(d, mh, sz, cudaMemcpyHostToDevice);
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaMemcpy(d, mh, sz, cudaMemcpyHostToDevice);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  cudaMallocHost memcpy: %.2f us = %.1f GB/s\n",
               best, sz/(best/1e6)/1e9);
    }
    cudaFreeHost(mh);

    // HostRegister on malloc
    void *hr = malloc(sz);
    memset(hr, 0, sz);
    cudaHostRegister(hr, sz, cudaHostRegisterDefault);
    {
        for (int i = 0; i < 3; i++) cudaMemcpy(d, hr, sz, cudaMemcpyHostToDevice);
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaMemcpy(d, hr, sz, cudaMemcpyHostToDevice);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  cudaHostRegister memcpy: %.2f us = %.1f GB/s\n",
               best, sz/(best/1e6)/1e9);
    }
    cudaHostUnregister(hr);
    free(hr);

    cudaFree(d);
    return 0;
}
