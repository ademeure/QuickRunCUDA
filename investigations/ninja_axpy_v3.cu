// L1 axpy v3 — RANDOM data + ncu-comparable + 3 measurement methods
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

__global__ void fill_random(__nv_bfloat16 *p, int N, unsigned seed) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    unsigned x = tid * 2654435761u + seed;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    p[tid] = __float2bfloat16((float)((x & 0xffff) - 0x8000) / 32768.0f);
}

__launch_bounds__(256, 4) __global__ void axpy_v1(
    const __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ y, float a, int N)
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

    int blocks = 131072;
    
    // Test 1: ZERO data (cudaMemset 0)
    cudaMemset(d_x, 0, bytes); cudaMemset(d_y, 0, bytes);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 5; i++) axpy_v1<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
    cudaDeviceSynchronize();
    float best_zero = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        axpy_v1<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_zero) best_zero = ms;
    }
    double gbs_zero = (bytes * 3.0) / (best_zero/1000) / 1e9;
    printf("ZERO data:    %.4f ms = %.0f GB/s = %.2f%% of 7672 spec\n", best_zero, gbs_zero, gbs_zero/7672*100);

    // Test 2: 0x42 constant
    cudaMemset(d_x, 0x42, bytes); cudaMemset(d_y, 0x33, bytes);
    for (int i = 0; i < 5; i++) axpy_v1<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
    cudaDeviceSynchronize();
    float best_const = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        axpy_v1<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_const) best_const = ms;
    }
    double gbs_const = (bytes * 3.0) / (best_const/1000) / 1e9;
    printf("CONSTANT data: %.4f ms = %.0f GB/s = %.2f%% of 7672 spec\n", best_const, gbs_const, gbs_const/7672*100);

    // Test 3: RANDOM data
    fill_random<<<(N + 255) / 256, 256>>>(d_x, N, 42);
    fill_random<<<(N + 255) / 256, 256>>>(d_y, N, 84);
    cudaDeviceSynchronize();
    for (int i = 0; i < 5; i++) axpy_v1<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
    cudaDeviceSynchronize();
    float best_rand = 1e30f;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(e0);
        axpy_v1<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_rand) best_rand = ms;
    }
    double gbs_rand = (bytes * 3.0) / (best_rand/1000) / 1e9;
    printf("RANDOM data:  %.4f ms = %.0f GB/s = %.2f%% of 7672 spec\n", best_rand, gbs_rand, gbs_rand/7672*100);
    return 0;
}
