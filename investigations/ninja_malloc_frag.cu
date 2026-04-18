// C2: cudaMallocAsync vs cudaMalloc allocation latency + fragmentation
//
// Test: 1) latency per call; 2) sustained alloc/free over many cycles
// Hypothesis: cudaMallocAsync pool-based, much faster repeated alloc/free
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    // === Test 1: single-call latency at fresh state ===
    printf("# Single-call latency (fresh state)\n");
    for (size_t bytes : {1024UL, 1024*1024UL, 16*1024*1024UL, 1024*1024*1024UL}) {
        // cudaMalloc
        void *p; auto t0 = std::chrono::high_resolution_clock::now();
        cudaMalloc(&p, bytes);
        auto t1 = std::chrono::high_resolution_clock::now();
        cudaFree(p);
        // cudaMallocAsync
        auto t2 = std::chrono::high_resolution_clock::now();
        cudaMallocAsync(&p, bytes, s);
        cudaStreamSynchronize(s);
        auto t3 = std::chrono::high_resolution_clock::now();
        cudaFreeAsync(p, s);
        cudaStreamSynchronize(s);
        printf("  size=%5ld MB  cudaMalloc: %.1f us  cudaMallocAsync: %.1f us\n",
               bytes / (1024*1024),
               std::chrono::duration<double, std::micro>(t1-t0).count(),
               std::chrono::duration<double, std::micro>(t3-t2).count());
    }

    // === Test 2: Sustained alloc/free, measure mean latency over 1000 cycles ===
    printf("\n# Sustained alloc/free (1000 cycles, 16 MB each)\n");
    {
        size_t bytes = 16 * 1024 * 1024;
        // cudaMalloc baseline
        double total_malloc = 0;
        std::vector<void*> ptrs;
        ptrs.reserve(1000);
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; i++) {
            void *p; cudaMalloc(&p, bytes); ptrs.push_back(p);
        }
        for (auto p : ptrs) cudaFree(p);
        auto t1 = std::chrono::high_resolution_clock::now();
        total_malloc = std::chrono::duration<double, std::micro>(t1-t0).count();
        printf("  cudaMalloc 1000× alloc + 1000× free:      %.1f ms  (%.1f us/op)\n",
               total_malloc/1000, total_malloc/2000);

        // cudaMallocAsync — first run (warmup pool)
        ptrs.clear();
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; i++) {
            void *p; cudaMallocAsync(&p, bytes, s); ptrs.push_back(p);
        }
        for (auto p : ptrs) cudaFreeAsync(p, s);
        cudaStreamSynchronize(s);
        auto t3 = std::chrono::high_resolution_clock::now();
        double total_async1 = std::chrono::duration<double, std::micro>(t3-t2).count();
        printf("  cudaMallocAsync 1000× (1st run, pool warm): %.1f ms  (%.1f us/op)\n",
               total_async1/1000, total_async1/2000);

        // cudaMallocAsync — second run (pool already warm)
        ptrs.clear();
        auto t4 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < 1000; i++) {
            void *p; cudaMallocAsync(&p, bytes, s); ptrs.push_back(p);
        }
        for (auto p : ptrs) cudaFreeAsync(p, s);
        cudaStreamSynchronize(s);
        auto t5 = std::chrono::high_resolution_clock::now();
        double total_async2 = std::chrono::duration<double, std::micro>(t5-t4).count();
        printf("  cudaMallocAsync 1000× (2nd run, pool warm): %.1f ms  (%.1f us/op)\n",
               total_async2/1000, total_async2/2000);
    }

    // === Test 3: Fragmentation — alternating big/small allocs ===
    printf("\n# Fragmentation test: alternating 1 MB + 100 MB, 100 cycles\n");
    {
        size_t small = 1024 * 1024, large = 100 * 1024 * 1024;
        // cudaMalloc
        size_t free_b, total_b;
        cudaMemGetInfo(&free_b, &total_b);
        printf("  Initial free: %ld MB\n", free_b / (1024*1024));
        std::vector<void*> ptrs;
        for (int i = 0; i < 100; i++) {
            void *p1, *p2;
            cudaMalloc(&p1, small);  ptrs.push_back(p1);
            cudaMalloc(&p2, large);  cudaFree(p2);  // free large immediately
        }
        cudaMemGetInfo(&free_b, &total_b);
        printf("  After 100× cudaMalloc small+large/free large:  free=%ld MB\n", free_b / (1024*1024));
        // Try big alloc
        void *big; cudaError_t err = cudaMalloc(&big, 1024 * 1024 * 1024);
        printf("  cudaMalloc 1 GB after fragmentation: %s\n", err == cudaSuccess ? "SUCCESS" : cudaGetErrorString(err));
        if (err == cudaSuccess) cudaFree(big);
        for (auto p : ptrs) cudaFree(p);
    }
    return 0;
}
