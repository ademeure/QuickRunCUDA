// NINJA histogram: optimize block count and per-warp work
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

// Same SMEM aggregation but tunable block count
__launch_bounds__(256, 4) __global__ void hist_ninja(
    const __nv_bfloat16 *x, unsigned *bins, int N, int per_warp_elems)
{
    __shared__ unsigned smem_bins[256];
    int t = threadIdx.x;
    if (t < 256) smem_bins[t] = 0;
    __syncthreads();

    int warp = t >> 5;
    int lane = t & 31;
    int warp_global = blockIdx.x * 8 + warp;
    int total_warps = gridDim.x * 8;

    // Each warp handles per_warp_elems consecutive elements (in chunks of 8 BF16)
    int elems_per_iter = 256;  // 32 lanes × 8 elems
    for (int wbase = warp_global * per_warp_elems;
         wbase < N;
         wbase += total_warps * per_warp_elems) {
        for (int off = 0; off < per_warp_elems && wbase + off < N; off += elems_per_iter) {
            int my_off = wbase + off + lane * 8;
            if (my_off + 7 >= N) continue;
            uint4 v = *(uint4*)&x[my_off];
            const unsigned short *p = (const unsigned short*)&v;
            #pragma unroll
            for (int j = 0; j < 8; j++) {
                unsigned b = (p[j] >> 7) & 0xFF;
                atomicAdd_block(&smem_bins[b], 1u);
            }
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

    auto bench = [&](int blocks, int per_warp) {
        cudaMemset(d_bins, 0, 256 * 4);
        for (int i = 0; i < 5; i++) {
            cudaMemset(d_bins, 0, 256 * 4);
            hist_ninja<<<blocks, 256>>>(d_x, d_bins, N, per_warp);
        }
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 20; i++) {
            cudaMemset(d_bins, 0, 256 * 4);
            cudaEventRecord(e0);
            hist_ninja<<<blocks, 256>>>(d_x, d_bins, N, per_warp);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        unsigned hb[256]; cudaMemcpy(hb, d_bins, 256*4, cudaMemcpyDeviceToHost);
        long total = 0; for (int i = 0; i < 256; i++) total += hb[i];
        double gbs = bytes / (best/1000) / 1e9;
        printf("  blocks=%5d per_warp=%6d: %.4f ms = %.1f GB/s = %.2f%% spec  total=%ld\n",
               blocks, per_warp, best, gbs, gbs/7672*100, total);
    };

    printf("# tune block count + per-warp work\n");
    for (int b : {148, 296, 592, 1184, 2368, 4736, 9472}) {
        for (int pw : {256, 1024, 4096, 16384}) {
            bench(b, pw);
        }
    }
    return 0;
}
