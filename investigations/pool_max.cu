// cudaMallocAsync pool size limits and behavior
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);

    // Get pool attributes
    size_t threshold, used, used_high, reserved, reserved_high;
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemHigh, &used_high);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemHigh, &reserved_high);

    printf("# Default pool initial state:\n");
    printf("  ReleaseThreshold: %zu B\n", threshold);
    printf("  UsedMem current: %zu B\n", used);
    printf("  ReservedMem current: %zu B\n", reserved);

    // Allocate progressively larger amounts
    printf("\n# Pool growth pattern\n");
    printf("# %-12s %-15s %-15s\n", "alloc_GB", "alloc_us", "reserved_GB");
    std::vector<void*> ptrs;
    for (size_t gb : {1ul, 2ul, 4ul, 8ul, 16ul, 32ul, 64ul, 128ul}) {
        size_t bytes = gb * 1024 * 1024 * 1024;
        auto t0 = std::chrono::high_resolution_clock::now();
        void *p;
        cudaError_t err = cudaMallocAsync(&p, bytes, s);
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        if (err) {
            printf("  %-12zu FAIL: %s\n", gb, cudaGetErrorString(err));
            break;
        }
        ptrs.push_back(p);
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
        printf("  %-12zu %-15.0f %-15.2f\n", gb, us, reserved/1e9);
    }

    // Free all
    for (auto p : ptrs) cudaFreeAsync(p, s);
    cudaStreamSynchronize(s);

    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved);
    cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used);
    printf("\n  After free: reserved=%.2f GB, used=%zu B\n", reserved/1e9, used);

    // Try to alloc the largest possible (should still get the pool)
    auto t0 = std::chrono::high_resolution_clock::now();
    void *huge;
    cudaError_t err = cudaMallocAsync(&huge, 100ul*1024*1024*1024, s);
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    float us = std::chrono::duration<float, std::micro>(t1-t0).count();
    printf("\n  100 GB realloc: %s (%.0f us)\n",
           err ? cudaGetErrorString(err) : "ok", us);

    return 0;
}
