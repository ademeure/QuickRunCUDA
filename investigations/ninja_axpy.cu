// NINJA axpy: tune block count
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

__launch_bounds__(256, 4) __global__ void axpy_orig(
    const __nv_bfloat16 *x, __nv_bfloat16 *y, float a, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
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
        for (int j = 0; j < 8; j++) {
            yb[j] = __hadd(__hmul(xb[j], a_bf), yb[j]);
        }
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
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](int blocks) {
        for (int i = 0; i < 3; i++) axpy_orig<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaEventRecord(e0);
            axpy_orig<<<blocks, 256>>>(d_x, d_y, 0.5f, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        size_t total = bytes * 3;  // R(x) + R(y) + W(y)
        double gbs = total / (best/1000) / 1e9;
        printf("  blocks=%5d: %.4f ms = %.0f GB/s = %.2f%% spec\n",
               blocks, best, gbs, gbs/7672*100);
    };

    for (int b : {148, 296, 444, 592, 740, 888, 1184, 1480, 2368, 4736, 9472, 18944, 37888, 65536, 131072}) bench(b);
    return 0;
}
