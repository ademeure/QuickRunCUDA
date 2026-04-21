// V8 H4: UMA page migration cost
// MODE 0: managed mem touched only on host (GPU page-faults on access)
// MODE 1: managed mem prefetched to GPU first
// MODE 2: managed mem with cudaMemAdvise SetReadMostly
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void touch(float* buf, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    float sum = 0;
    for (int i = gtid; i < n; i += total) sum += buf[i];
    if (sum == 1.234567e-30f) buf[0] = sum;
}

int main() {
    cudaSetDevice(0);
    size_t SIZE = 64 * 1024 * 1024;  // 64 MB
    int N = SIZE / 4;

    cudaStream_t s; cudaStreamCreate(&s);

    // Test 1: cold managed memory — first touch from GPU
    {
        float* buf;
        cudaMallocManaged(&buf, SIZE);
        for (int i = 0; i < N; i++) buf[i] = (float)i;  // host-only init
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        touch<<<148, 256, 0, s>>>(buf, N);
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double cold_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        // Subsequent touch (data now resident on GPU)
        auto t2 = std::chrono::high_resolution_clock::now();
        touch<<<148, 256, 0, s>>>(buf, N);
        cudaStreamSynchronize(s);
        auto t3 = std::chrono::high_resolution_clock::now();
        double warm_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

        printf("UMA cold (host-init, GPU faults):  %.0f us\n", cold_us);
        printf("UMA warm (already on GPU):         %.0f us\n", warm_us);
        printf("Migration overhead: %.0f us\n", cold_us - warm_us);
        cudaFree(buf);
    }

    // Test 2: with prefetch
    {
        float* buf;
        cudaMallocManaged(&buf, SIZE);
        for (int i = 0; i < N; i++) buf[i] = (float)i;
        cudaMemPrefetchAsync(buf, SIZE, cudaMemLocation{cudaMemLocationTypeDevice, 0}, 0, s);
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::high_resolution_clock::now();
        touch<<<148, 256, 0, s>>>(buf, N);
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        printf("UMA with prefetch:                 %.0f us\n",
               std::chrono::duration<double, std::micro>(t1 - t0).count());
        cudaFree(buf);
    }

    return 0;
}
