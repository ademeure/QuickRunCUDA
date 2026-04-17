// L2 cache line / sectorization investigation
// Detect cache line size by stride access pattern
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 1000

extern "C" __global__ void stride_access(float *src, float *out, int N, int stride_bytes) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride_floats = stride_bytes / 4;
    int total_threads = gridDim.x * blockDim.x;
    float acc = 0;

    // Each thread reads N elements with given stride
    // Multiple threads access spaced addresses
    int base = tid * stride_floats;
    for (int i = 0; i < N; i++) {
        int idx = (base + i * stride_floats * total_threads) & ((64 << 20) - 1);  // 256 MB wrap
        acc += src[idx];
    }
    if (acc < -1e30f) out[tid] = acc;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    int N = 64 << 20;  // 256 MB
    float *d_src, *d_out;
    cudaMalloc(&d_src, N * sizeof(float));
    cudaMalloc(&d_out, prop.multiProcessorCount * 256 * sizeof(float));
    cudaMemset(d_src, 0x40, N * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 stride access pattern (148 blocks × 256 thr, %d reads/thread)\n", ITERS);
    printf("# %-12s %-12s\n", "stride_bytes", "BW_GB/s (effective)");

    int strides[] = {4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};
    for (int sb : strides) {
        float t = bench([&]{
            stride_access<<<prop.multiProcessorCount, 256, 0, s>>>(d_src, d_out, ITERS, sb);
        });
        size_t accesses = (size_t)prop.multiProcessorCount * 256 * ITERS;
        // Each access fetches at least 32 B (warp granularity) but uses only 4 B
        float useful_bw = accesses * 4 / (t/1e3) / 1e9;
        float total_fetched = accesses * 32 / (t/1e3) / 1e9;  // assuming 32B sector
        printf("  %-12d %-12.1f (effective; fetched ~%.1f GB/s assuming 32B sectors)\n",
               sb, useful_bw, total_fetched);
    }

    cudaStreamDestroy(s);
    cudaFree(d_src); cudaFree(d_out);
    return 0;
}
