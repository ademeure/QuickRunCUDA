// Histogram tuning: minor variants of original L3 recipe
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

// Original L3 recipe (commit 492a5f6)
__launch_bounds__(256, 4) __global__ void hist_orig(
    const __nv_bfloat16 *x, unsigned *bins, int N)
{
    __shared__ unsigned smem_bins[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    if (t < 256) smem_bins[t] = 0;
    __syncthreads();
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;
    int lane = threadIdx.x & 31;
    for (int i = warp_base + lane * 8; i < N - 7; i += stride * 8) {
        uint4 v = *(uint4*)&x[i];
        const unsigned short *p = (const unsigned short*)&v;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned b = (p[j] >> 7) & 0xFF;
            atomicAdd_block(&smem_bins[b], 1u);
        }
    }
    __syncthreads();
    if (t < 256) atomicAdd(&bins[t], smem_bins[t]);
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 1ull * 1024 * 1024 * 1024;
    int N = bytes / sizeof(__nv_bfloat16);
    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) h[i] = __float2bfloat16(((float)rand()/RAND_MAX) * 2 - 1);
    __nv_bfloat16 *d_x; cudaMalloc(&d_x, bytes);
    cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice);
    free(h);
    unsigned *d_bins; cudaMalloc(&d_bins, 256 * sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](int blocks) {
        cudaMemset(d_bins, 0, 256 * 4);
        for (int i = 0; i < 5; i++) {
            cudaMemset(d_bins, 0, 256 * 4);
            hist_orig<<<blocks, 256>>>(d_x, d_bins, N);
        }
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaMemset(d_bins, 0, 256 * 4);
            cudaEventRecord(e0);
            hist_orig<<<blocks, 256>>>(d_x, d_bins, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double gbs = bytes / (best/1000) / 1e9;
        printf("  blocks=%5d: %.4f ms = %.1f GB/s = %.2f%% spec\n",
               blocks, best, gbs, gbs/7672*100);
    };

    for (int b : {148, 296, 444, 592, 740, 888, 1184, 1480, 1776, 2368, 4736, 9472}) bench(b);
    return 0;
}
