// NINJA hist v3: register-keep-alive trick + tuned block count
// Currently at 87.4% spec; HBM read SoL = 95.3%. Can we close the gap?
//
// The trick: each thread loads uint4, processes WITHIN registers, then
// does ONE shared-memory atomic per element. No re-loads.
//
// Already does this in current best version. So the bottleneck is...
// the atomic? Let me look at SMEM atomic throughput.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

// V1: original-style with stride loop + per-warp coalesced
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

// V2: persistent-style — each warp processes a fixed chunk
__launch_bounds__(256, 4) __global__ void hist_chunked(
    const __nv_bfloat16 *x, unsigned *bins, int N)
{
    __shared__ unsigned smem_bins[256];
    int t = threadIdx.x;
    if (t < 256) smem_bins[t] = 0;
    __syncthreads();
    int warp_in_grid = blockIdx.x * 8 + (t >> 5);
    int total_warps = gridDim.x * 8;
    int lane = t & 31;
    int chunk = (N + total_warps - 1) / total_warps;
    int my_start = warp_in_grid * chunk;
    int my_end = min(my_start + chunk, N);
    for (int i = my_start + lane * 8; i + 7 < my_end; i += 256) {
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

// V3: privatized bins per warp (no atomic during processing, atomic only at end)
__launch_bounds__(256, 4) __global__ void hist_privatized(
    const __nv_bfloat16 *x, unsigned *bins, int N)
{
    // Each of 8 warps gets its own 256-bin counter array
    __shared__ unsigned smem_bins[8 * 256];  // 8 KB
    int t = threadIdx.x;
    int warp = t >> 5;
    int lane = t & 31;
    // Zero
    #pragma unroll
    for (int i = t; i < 8 * 256; i += blockDim.x) smem_bins[i] = 0;
    __syncthreads();

    int warp_in_grid = blockIdx.x * 8 + warp;
    int total_warps = gridDim.x * 8;
    int chunk = (N + total_warps - 1) / total_warps;
    int my_start = warp_in_grid * chunk;
    int my_end = min(my_start + chunk, N);
    for (int i = my_start + lane * 8; i + 7 < my_end; i += 256) {
        uint4 v = *(uint4*)&x[i];
        const unsigned short *p = (const unsigned short*)&v;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned b = (p[j] >> 7) & 0xFF;
            atomicAdd_block(&smem_bins[warp * 256 + b], 1u);  // per-warp slot, less contention
        }
    }
    __syncthreads();
    // Reduce 8 copies to 1, then global atomic
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

    auto bench = [&](auto launch, int blocks, const char* label) {
        cudaMemset(d_bins, 0, 256 * 4);
        for (int i = 0; i < 5; i++) {
            cudaMemset(d_bins, 0, 256 * 4);
            launch();
        }
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 30; i++) {
            cudaMemset(d_bins, 0, 256 * 4);
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        unsigned hb[256]; cudaMemcpy(hb, d_bins, 256*4, cudaMemcpyDeviceToHost);
        long total = 0; for (int i = 0; i < 256; i++) total += hb[i];
        double gbs = bytes / (best/1000) / 1e9;
        printf("  blocks=%5d %s: %.4f ms = %.0f GB/s = %.2f%% spec total=%ld\n",
               blocks, label, best, gbs, gbs/7672*100, total);
    };

    for (int b : {296, 444, 592, 740, 888, 1184, 2368}) {
        bench([&]{ hist_orig<<<b, 256>>>(d_x, d_bins, N); }, b, "orig    ");
        bench([&]{ hist_chunked<<<b, 256>>>(d_x, d_bins, N); }, b, "chunked ");
        bench([&]{ hist_privatized<<<b, 256, 0>>>(d_x, d_bins, N); }, b, "private8");
    }
    return 0;
}
