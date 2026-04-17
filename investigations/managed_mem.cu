// Managed memory page migration cost on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstring>

extern "C" __global__ void touch_kernel(int *ptr, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int sum = 0;
    for (int i = tid; i < N; i += stride) {
        sum += ptr[i];
    }
    if (sum < -1e9) ptr[tid] = sum;
}

int main() {
    cudaSetDevice(0);

    printf("# B300 managed memory migration cost\n\n");

    // Allocate managed memory
    size_t bytes = 64 * 1024 * 1024;  // 64 MB
    int *managed;
    cudaMallocManaged(&managed, bytes);
    int N = bytes / sizeof(int);

    // Test 1: Initialize on host (CPU page-faults), then access from GPU
    {
        memset(managed, 0, bytes);  // CPU touch — pages on host
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        touch_kernel<<<148, 256>>>(managed, N);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        printf("Cold migration host→GPU (64 MB): %.2f ms = %.1f GB/s\n",
               ms, bytes/(ms/1e3)/1e9);
    }

    // Test 2: Already on GPU - second access (warm)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        touch_kernel<<<148, 256>>>(managed, N);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        printf("Warm GPU access (already migrated): %.2f ms = %.1f GB/s\n",
               ms, bytes/(ms/1e3)/1e9);
    }

    // Test 3: Touch from CPU again (migrate back)
    {
        auto t0 = std::chrono::high_resolution_clock::now();
        memset(managed, 1, bytes);  // CPU writes
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        printf("CPU access (migrate back from GPU): %.2f ms = %.1f GB/s\n",
               ms, bytes/(ms/1e3)/1e9);
    }

    // Test 4: Use prefetch hint
    {
        cudaMemLocation loc;
        loc.type = cudaMemLocationTypeDevice;
        loc.id = 0;
        cudaMemPrefetchAsync(managed, bytes, loc, 0, 0);  // prefetch to GPU 0
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        touch_kernel<<<148, 256>>>(managed, N);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
        printf("After cudaMemPrefetchAsync(GPU): %.2f ms = %.1f GB/s\n",
               ms, bytes/(ms/1e3)/1e9);
    }

    // Test 5: Compare to plain cudaMalloc + cudaMemcpy
    {
        int *gpu;
        int *host = (int*)malloc(bytes);
        memset(host, 0, bytes);
        cudaMalloc(&gpu, bytes);

        // Warmup
        cudaMemcpy(gpu, host, bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        cudaMemcpy(gpu, host, bytes, cudaMemcpyHostToDevice);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms_h2d = std::chrono::duration<float, std::milli>(t1-t0).count();

        t0 = std::chrono::high_resolution_clock::now();
        touch_kernel<<<148, 256>>>(gpu, N);
        cudaDeviceSynchronize();
        t1 = std::chrono::high_resolution_clock::now();
        float ms_kernel = std::chrono::duration<float, std::milli>(t1-t0).count();

        printf("\nReference: plain cudaMalloc + cudaMemcpy(64MB) + kernel:\n");
        printf("  cudaMemcpy 64MB pageable: %.2f ms = %.1f GB/s\n", ms_h2d, bytes/(ms_h2d/1e3)/1e9);
        printf("  Kernel scan:              %.2f ms = %.1f GB/s\n", ms_kernel, bytes/(ms_kernel/1e3)/1e9);

        free(host);
        cudaFree(gpu);
    }

    cudaFree(managed);
    return 0;
}
