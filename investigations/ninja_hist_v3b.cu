// Clean rerun: hist_privatized only
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

__launch_bounds__(256, 4) __global__ void hist_priv8(
    const __nv_bfloat16 *x, unsigned *bins, int N)
{
    __shared__ unsigned smem_bins[8 * 256];
    int t = threadIdx.x;
    int warp = t >> 5;
    int lane = t & 31;
    #pragma unroll
    for (int i = t; i < 8 * 256; i += blockDim.x) smem_bins[i] = 0;
    __syncthreads();

    int warp_in_grid = blockIdx.x * 8 + warp;
    int total_warps = gridDim.x * 8;
    // Use stride loop with original L3 access pattern
    int stride = total_warps * 256;  // bytes processed per cycle
    int warp_base = warp_in_grid * 256;  // 32 lanes × 8 elems = 256 elems per warp per iter
    for (int i = warp_base + lane * 8; i + 7 < N; i += stride) {
        uint4 v = *(uint4*)&x[i];
        const unsigned short *p = (const unsigned short*)&v;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned b = (p[j] >> 7) & 0xFF;
            atomicAdd_block(&smem_bins[warp * 256 + b], 1u);
        }
    }
    __syncthreads();
    if (t < 256) {
        unsigned sum = 0;
        #pragma unroll
        for (int w = 0; w < 8; w++) sum += smem_bins[w * 256 + t];
        atomicAdd(&bins[t], sum);
    }
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
            hist_priv8<<<blocks, 256>>>(d_x, d_bins, N);
        }
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaMemset(d_bins, 0, 256 * 4);
            cudaEventRecord(e0);
            hist_priv8<<<blocks, 256>>>(d_x, d_bins, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        unsigned hb[256]; cudaMemcpy(hb, d_bins, 256*4, cudaMemcpyDeviceToHost);
        long total = 0; for (int i = 0; i < 256; i++) total += hb[i];
        double gbs = bytes / (best/1000) / 1e9;
        printf("  blocks=%5d: %.4f ms = %.0f GB/s = %.2f%% spec total=%ld\n",
               blocks, best, gbs, gbs/7672*100, total);
    };

    for (int b : {296, 444, 592, 740, 888, 1184, 2368, 4736, 9472}) bench(b);
    return 0;
}
