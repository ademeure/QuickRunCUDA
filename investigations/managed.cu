// Managed memory + cuMemAdvise: page migration costs
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void touch(float *data, int N, float v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) data[i] += v;
}

int main() {
    cudaSetDevice(0);

    int concurrent_managed;
    cudaDeviceGetAttribute(&concurrent_managed, cudaDevAttrConcurrentManagedAccess, 0);
    printf("# B300 ConcurrentManagedAccess: %d\n", concurrent_managed);

    int pageable_concurrent;
    cudaDeviceGetAttribute(&pageable_concurrent, cudaDevAttrPageableMemoryAccess, 0);
    printf("# PageableMemoryAccess: %d\n", pageable_concurrent);

    int pageable_uses_pt;
    cudaDeviceGetAttribute(&pageable_uses_pt, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0);
    printf("# PageableMemoryAccessUsesHostPageTables: %d\n", pageable_uses_pt);

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 2; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("\n# Managed memory: first-touch and migration costs\n");
    printf("# %-10s %-15s %-15s %-15s\n", "size_MB", "first_GPU_ms", "rewarm_GPU_ms", "GB/s");

    for (int sz_mb : {16, 64, 256, 1024}) {
        size_t bytes = (size_t)sz_mb * 1024 * 1024;
        size_t N = bytes / sizeof(float);

        float *m_data;
        cudaMallocManaged(&m_data, bytes);
        // Zero on CPU
        memset(m_data, 0, bytes);

        // First GPU access - migration cost
        auto t0 = std::chrono::high_resolution_clock::now();
        touch<<<148, 256>>>(m_data, N, 1.0f);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float first = std::chrono::duration<float, std::milli>(t1-t0).count();

        // Second access (data already on GPU)
        float warm = bench([&]{ touch<<<148, 256>>>(m_data, N, 1.0f); });

        float bw = bytes / (warm/1000) / 1e9;
        printf("  %-10d %-15.2f %-15.3f %-15.1f\n", sz_mb, first, warm, bw);

        cudaFree(m_data);
    }

    // cudaMemAdvise: ReadMostly hint
    printf("\n# cudaMemAdvise: PreferredLocation, ReadMostly\n");
    {
        size_t bytes = 256 * 1024 * 1024;
        size_t N = bytes / sizeof(float);
        float *m_data;
        cudaMallocManaged(&m_data, bytes);
        memset(m_data, 0, bytes);

        // Set ReadMostly hint - new API uses cudaMemLocation struct
        cudaMemLocation loc = {cudaMemLocationTypeDevice, 0};
        cudaMemAdvise(m_data, bytes, cudaMemAdviseSetReadMostly, loc);
        cudaMemAdvise(m_data, bytes, cudaMemAdviseSetPreferredLocation, loc);

        // Prefetch to GPU
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaMemPrefetchAsync(m_data, bytes, loc, 0, 0);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float prefetch_ms = std::chrono::duration<float, std::milli>(t1-t0).count();

        // GPU access (should be fast - no migration)
        float t_warm = bench([&]{ touch<<<148, 256>>>(m_data, N, 1.0f); });

        printf("  Prefetch 256 MB to GPU: %.2f ms = %.1f GB/s\n",
               prefetch_ms, bytes/(prefetch_ms/1000)/1e9);
        printf("  GPU access after prefetch: %.3f ms = %.1f GB/s\n",
               t_warm, bytes/(t_warm/1000)/1e9);

        // Migrate back to CPU
        cudaMemLocation cpu_loc = {cudaMemLocationTypeHost, 0};
        t0 = std::chrono::high_resolution_clock::now();
        cudaMemPrefetchAsync(m_data, bytes, cpu_loc, 0, 0);
        cudaDeviceSynchronize();
        t1 = std::chrono::high_resolution_clock::now();
        float back_ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        printf("  Prefetch 256 MB back to CPU: %.2f ms = %.1f GB/s\n",
               back_ms, bytes/(back_ms/1000)/1e9);

        cudaFree(m_data);
    }

    // PageableMemoryAccess - direct CPU memory access from GPU
    printf("\n# Direct pageable memory access from GPU (if supported)\n");
    if (pageable_concurrent) {
        size_t bytes = 64 * 1024 * 1024;
        size_t N = bytes / sizeof(float);
        float *h_data = (float*)malloc(bytes);
        memset(h_data, 0, bytes);

        // GPU directly reads/writes pageable host memory
        float t = bench([&]{ touch<<<148, 256>>>(h_data, N, 1.0f); });
        float bw = bytes / (t/1000) / 1e9;
        printf("  GPU touches 64 MB pageable host: %.2f ms = %.1f GB/s\n", t, bw);
        free(h_data);
    } else {
        printf("  Not supported (would crash)\n");
    }

    return 0;
}
