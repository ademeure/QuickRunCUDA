// CUDA memory pool allocation/free patterns and release threshold
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    // Get default pool
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials=1000) {
        for (int i = 0; i < 10; i++) fn();
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

    printf("# B300 cudaMallocAsync / cudaMemPool behavior\n\n");

    // Test 1: cudaMallocAsync vs cudaMalloc cost (small alloc)
    printf("## Test 1: Allocation cost vs size (best of 1000)\n");
    printf("  %-12s %-15s %-15s %-15s\n", "size", "cudaMalloc", "MallocAsync", "Ratio");
    for (size_t sz : {64ul, 4096ul, 65536ul, 1024ul*1024, 16ul*1024*1024, 256ul*1024*1024}) {
        // sync alloc/free
        float t_sync = bench([&]{
            void *p; cudaMalloc(&p, sz); cudaFree(p);
        }, 100);
        // async via pool
        float t_async = bench([&]{
            void *p; cudaMallocAsync(&p, sz, s); cudaFreeAsync(p, s);
        }, 100);
        printf("  %-12zu %-15.0f %-15.0f %.2fx\n",
               sz, t_sync, t_async, t_sync / t_async);
    }

    // Test 2: Pool reuse - alloc/free in a loop without sync
    printf("\n## Test 2: Same-size repeated alloc/free in same stream (pool reuse)\n");
    {
        size_t sz = 1024 * 1024;
        // Warmup
        for (int i = 0; i < 100; i++) {
            void *p; cudaMallocAsync(&p, sz, s); cudaFreeAsync(p, s);
        }
        cudaStreamSynchronize(s);

        float t = bench([&]{
            void *p; cudaMallocAsync(&p, sz, s); cudaFreeAsync(p, s);
        });
        printf("  Hot reuse (1MB): %.0f ns per cycle\n", t);
    }

    // Test 3: Release threshold behavior
    printf("\n## Test 3: Release threshold behavior\n");
    {
        size_t threshold;
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
        printf("  Default release threshold: %zu B\n", threshold);

        size_t reserved_mem, used_mem;
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_mem);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrUsedMemCurrent, &used_mem);
        printf("  Reserved: %zu B, Used: %zu B\n", reserved_mem, used_mem);

        // Set release threshold to UINT64_MAX (never release)
        size_t big = (size_t)-1;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &big);

        // Allocate then free, check reserved
        void *p; cudaMallocAsync(&p, 256*1024*1024, s); cudaFreeAsync(p, s);
        cudaStreamSynchronize(s);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_mem);
        printf("  After 256MB alloc/free with infinite threshold: reserved=%zu B\n", reserved_mem);

        // Trim
        cudaMemPoolTrimTo(pool, 0);
        cudaMemPoolGetAttribute(pool, cudaMemPoolAttrReservedMemCurrent, &reserved_mem);
        printf("  After TrimTo(0): reserved=%zu B\n", reserved_mem);
    }

    // Test 4: Different sizes interleaved
    printf("\n## Test 4: Pool fragmentation: alternating sizes\n");
    {
        // Set high threshold so pool keeps memory
        size_t big = 1024ul * 1024 * 1024 * 4;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &big);

        float t_alt = bench([&]{
            void *a, *b, *c;
            cudaMallocAsync(&a, 1024, s);
            cudaMallocAsync(&b, 65536, s);
            cudaMallocAsync(&c, 1024*1024, s);
            cudaFreeAsync(c, s);
            cudaFreeAsync(b, s);
            cudaFreeAsync(a, s);
        }, 100);
        printf("  3 alloc/free cycle: %.0f ns total = %.0f ns per op\n", t_alt, t_alt/6);
    }

    // Test 5: Cross-stream sharing
    printf("\n## Test 5: Cross-stream alloc with stream-ordered free\n");
    {
        cudaStream_t s2; cudaStreamCreate(&s2);
        float t = bench([&]{
            void *p; cudaMallocAsync(&p, 4096, s);
            cudaFreeAsync(p, s2);  // free on different stream
        }, 100);
        printf("  Cross-stream free: %.0f ns\n", t);
        cudaStreamDestroy(s2);
    }

    return 0;
}
