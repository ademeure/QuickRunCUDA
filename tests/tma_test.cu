// TMA (Tensor Memory Accelerator) basic test on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;
using barrier = cuda::barrier<cuda::thread_scope_block>;

// Simple async memcpy via cooperative groups (pre-TMA)
extern "C" __global__ void k_memcpy_async(float *src, float *dst, int N) {
    extern __shared__ float buf[];
    auto block = cg::this_thread_block();

    // Async copy from global to shared
    cg::memcpy_async(block, buf, src, sizeof(float) * N);
    cg::wait(block);

    // Just touch result
    if (threadIdx.x == 0) dst[blockIdx.x] = buf[0] + buf[N-1];
}

// Sync memcpy for comparison
extern "C" __global__ void k_memcpy_sync(float *src, float *dst, int N) {
    extern __shared__ float buf[];
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) buf[i] = src[i];
    __syncthreads();
    if (tid == 0) dst[blockIdx.x] = buf[0] + buf[N-1];
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    int N = 1024;  // 4 KB shmem
    float *d_src, *d_dst;
    cudaMalloc(&d_src, prop.multiProcessorCount * N * sizeof(float));
    cudaMalloc(&d_dst, prop.multiProcessorCount * sizeof(float));
    cudaMemset(d_src, 0x40, prop.multiProcessorCount * N * sizeof(float));

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

    int blocks = prop.multiProcessorCount;
    int threads = 128;
    int shmem_bytes = N * sizeof(float);

    printf("# B300 cuda::memcpy_async (pre-TMA cooperative groups async copy)\n");
    printf("# Loading %d floats (%d bytes) from GMEM to SHMEM per block\n\n", N, shmem_bytes);

    float t_async = bench([&]{
        k_memcpy_async<<<blocks, threads, shmem_bytes, s>>>(d_src, d_dst, N);
    });
    float t_sync = bench([&]{
        k_memcpy_sync<<<blocks, threads, shmem_bytes, s>>>(d_src, d_dst, N);
    });

    printf("  cg::memcpy_async: %.4f ms\n", t_async);
    printf("  manual sync copy: %.4f ms\n", t_sync);
    printf("  diff: %+.4f ms\n", t_async - t_sync);

    cudaStreamDestroy(s);
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
