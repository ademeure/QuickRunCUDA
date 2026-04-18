// Free rein L1: optimal axpy (y = a*x + y) at HBM peak
// THEORETICAL: 1 read of x, 1 RMW of y = 3 bytes/elem of HBM
// For BF16 1 GB x and 1 GB y: 3 GB traffic. At 7.30 TB/s peak: 411 us min.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

extern "C" __launch_bounds__(256, 4) __global__ void axpy_v8(
    const __nv_bfloat16 *x, __nv_bfloat16 *y, float a, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;
    int lane = threadIdx.x & 31;
    __nv_bfloat16 a_bf = __float2bfloat16(a);
    for (int i = warp_base + lane * 8; i < N - 7; i += stride * 8) {
        // Read x[i..i+7] and y[i..i+7] as uint4
        uint4 xv = *(uint4*)&x[i];
        uint4 yv = *(uint4*)&y[i];
        const __nv_bfloat16 *xb = (const __nv_bfloat16*)&xv;
        __nv_bfloat16 *yb = (__nv_bfloat16*)&yv;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            yb[j] = __hadd(__hmul(xb[j], a_bf), yb[j]);
        }
        // Write back
        *(uint4*)&y[i] = yv;
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 1ull * 1024 * 1024 * 1024;
    int N = bytes / sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    cudaMemset(d_x, 0x42, bytes);
    cudaMemset(d_y, 0x33, bytes);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 4, threads = 256;

    for (int i = 0; i < 3; i++) axpy_v8<<<blocks, threads>>>(d_x, d_y, 0.5f, N);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        axpy_v8<<<blocks, threads>>>(d_x, d_y, 0.5f, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    // Bytes touched: 1× R(x) + 1× R(y) + 1× W(y) = 3 GB
    size_t total_bytes = bytes * 3;
    double gbs = total_bytes / (best/1000) / 1e9;
    printf("# axpy_v8: %.4f ms = %.0f GB/s = %.1f%% of 7300 HBM peak\n",
           best, gbs, gbs/7300*100);
    return 0;
}
