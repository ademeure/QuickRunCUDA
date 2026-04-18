// Persistent kernel HBM saturation test
//
// Theoretical:
//   Per-launch axpy: 7.03 TB/s = AT SoL for 2:1 RW (commit 8aa9149)
//   Persistent kernel that loops over the same data should match if no overhead.
//   But persistent kernels have:
//     - per-iteration sync overhead
//     - grid-stride loop complexity
//     - command queue check (if checking for new commands)
//
// Method: launch persistent kernel ONCE, do K iterations of full axpy on the
// same buffer, signal CPU when all done.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <chrono>

extern "C" __launch_bounds__(256, 4) __global__ void persist_axpy_loop(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ y,
    float a,
    int N,
    int K_iters,
    int *flag)
{
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;
    int lane = threadIdx.x & 31;
    __nv_bfloat16 a_bf = __float2bfloat16(a);

    for (int k = 0; k < K_iters; k++) {
        for (int i = warp_base + lane * 8; i < N - 7; i += stride * 8) {
            uint4 xv = *(uint4*)&x[i];
            uint4 yv = *(uint4*)&y[i];
            const __nv_bfloat16 *xb = (const __nv_bfloat16*)&xv;
            __nv_bfloat16 *yb = (__nv_bfloat16*)&yv;
            #pragma unroll
            for (int j = 0; j < 8; j++) yb[j] = __hadd(__hmul(xb[j], a_bf), yb[j]);
            *(uint4*)&y[i] = yv;
        }
        // Could add cooperative_groups::this_grid().sync() between iters
        // but that adds overhead. For sat test, skip.
    }
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicExch(flag, 1);
}

extern "C" __launch_bounds__(256, 4) __global__ void single_axpy(
    const __nv_bfloat16 *__restrict__ x,
    __nv_bfloat16 *__restrict__ y,
    float a,
    int N)
{
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;
    int lane = threadIdx.x & 31;
    __nv_bfloat16 a_bf = __float2bfloat16(a);
    for (int i = warp_base + lane * 8; i < N - 7; i += stride * 8) {
        uint4 xv = *(uint4*)&x[i];
        uint4 yv = *(uint4*)&y[i];
        const __nv_bfloat16 *xb = (const __nv_bfloat16*)&xv;
        __nv_bfloat16 *yb = (__nv_bfloat16*)&yv;
        #pragma unroll
        for (int j = 0; j < 8; j++) yb[j] = __hadd(__hmul(xb[j], a_bf), yb[j]);
        *(uint4*)&y[i] = yv;
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 1ull * 1024 * 1024 * 1024;
    int N = bytes / sizeof(__nv_bfloat16);
    __nv_bfloat16 *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemset(d_x, 0x42, bytes); cudaMemset(d_y, 0x33, bytes);
    int *d_flag; cudaMalloc(&d_flag, 4);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = 131072;  // optimal from L1 axpy

    // Test 1: Per-launch axpy (baseline)
    for (int i = 0; i < 3; i++) single_axpy<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
    cudaDeviceSynchronize();
    float best1 = 1e30f;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(e0);
        single_axpy<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best1) best1 = ms;
    }
    double gbs1 = (bytes * 3.0) / (best1/1000.0) / 1e9;
    printf("Per-launch axpy:  %.4f ms = %.0f GB/s = %.2f%% spec\n", best1, gbs1, gbs1/7672*100);

    // Test 2: Persistent kernel doing K=20 iters of axpy
    int K_iters = 20;
    cudaMemset(d_flag, 0, 4);

    // Warmup
    persist_axpy_loop<<<blocks, 256>>>(d_x, d_y, 0.5f, N, 3, d_flag);
    cudaDeviceSynchronize();

    cudaMemset(d_flag, 0, 4);
    cudaEventRecord(e0);
    persist_axpy_loop<<<blocks, 256>>>(d_x, d_y, 0.5f, N, K_iters, d_flag);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms2; cudaEventElapsedTime(&ms2, e0, e1);
    double gbs2 = (bytes * 3.0 * K_iters) / (ms2/1000.0) / 1e9;
    double per_iter_ms = ms2 / K_iters;
    printf("Persistent K=%d:  %.4f ms total = %.4f ms/iter = %.0f GB/s = %.2f%% spec\n",
           K_iters, ms2, per_iter_ms, gbs2, gbs2/7672*100);

    // Test 3: K=100 iters
    K_iters = 100;
    cudaEventRecord(e0);
    persist_axpy_loop<<<blocks, 256>>>(d_x, d_y, 0.5f, N, K_iters, d_flag);
    cudaEventRecord(e1); cudaEventSynchronize(e1);
    float ms3; cudaEventElapsedTime(&ms3, e0, e1);
    double gbs3 = (bytes * 3.0 * K_iters) / (ms3/1000.0) / 1e9;
    double per_iter_ms3 = ms3 / K_iters;
    printf("Persistent K=%d: %.4f ms total = %.4f ms/iter = %.0f GB/s = %.2f%% spec\n",
           K_iters, ms3, per_iter_ms3, gbs3, gbs3/7672*100);

    return 0;
}
