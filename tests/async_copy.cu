// Async copy (cp.async) bandwidth test
#include <cuda_runtime.h>
#include <cuda/barrier>
#include <cuda/std/utility>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>
#include <cstdio>
#include <chrono>

namespace cg = cooperative_groups;

extern "C" __global__ void k_async(float *src, float *out, int N) {
    extern __shared__ float buf[];
    auto block = cg::this_thread_block();

    cg::memcpy_async(block, buf, src + blockIdx.x * N, sizeof(float) * N);
    cg::wait(block);

    if (threadIdx.x == 0) out[blockIdx.x] = buf[0];
}

extern "C" __global__ void k_sync_copy(float *src, float *out, int N) {
    extern __shared__ float buf[];
    int tid = threadIdx.x;
    for (int i = tid; i < N; i += blockDim.x) {
        buf[i] = src[blockIdx.x * N + i];
    }
    __syncthreads();
    if (tid == 0) out[blockIdx.x] = buf[0];
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount, threads = 256;

    int N = 1024;  // 4 KB per block
    float *d_src, *d_out;
    cudaMalloc(&d_src, blocks * N * sizeof(float));
    cudaMalloc(&d_out, blocks * sizeof(float));
    cudaMemset(d_src, 0x40, blocks * N * sizeof(float));

    cudaStream_t s; cudaStreamCreate(&s);
    cudaFuncSetAttribute((void*)k_async, cudaFuncAttributeMaxDynamicSharedMemorySize, N * 4);
    cudaFuncSetAttribute((void*)k_sync_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, N * 4);

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

    printf("# B300 async copy (cp.async) vs sync copy bandwidth\n");
    printf("# %d blocks × %d KB per block = %d KB total\n\n",
           blocks, N * 4 / 1024, blocks * N * 4 / 1024);

    int N_arr[] = {256, 1024, 4096, 16384};
    for (int n : N_arr) {
        if (n * 4 > 56 * 1024) continue;
        cudaFuncSetAttribute((void*)k_async, cudaFuncAttributeMaxDynamicSharedMemorySize, n * 4);
        cudaFuncSetAttribute((void*)k_sync_copy, cudaFuncAttributeMaxDynamicSharedMemorySize, n * 4);

        float t_async = bench([&]{
            k_async<<<blocks, threads, n * 4, s>>>(d_src, d_out, n);
        });
        float t_sync = bench([&]{
            k_sync_copy<<<blocks, threads, n * 4, s>>>(d_src, d_out, n);
        });

        size_t total_bytes = (size_t)blocks * n * 4;
        printf("  N=%-6d (%4d KB/block, %5d KB total):\n", n, n*4/1024, total_bytes/1024);
        printf("    async: %.4f ms, BW=%.1f GB/s\n",
               t_async, total_bytes/(t_async/1e3)/1e9);
        printf("    sync:  %.4f ms, BW=%.1f GB/s\n",
               t_sync, total_bytes/(t_sync/1e3)/1e9);
        printf("    speedup: %.2fx\n\n", t_sync / t_async);
    }

    cudaStreamDestroy(s);
    cudaFree(d_src); cudaFree(d_out);
    return 0;
}
