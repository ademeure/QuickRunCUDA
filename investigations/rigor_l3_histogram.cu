// L3 RIGOR: 256-bin histogram from BF16 exponent bits
//
// THEORETICAL: BF16 = 1 sign + 8 exp + 7 mantissa bits.
// Extract bits 14:7 (= exp + 1 mantissa MSB or just exp). With 8 bits = 256 bins.
// Read-bound at HBM peak 7.30 TB/s. 1 GB BF16 = 137 us minimum.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

extern "C" __launch_bounds__(256, 4) __global__ void hist_v1_naive(
    const __nv_bfloat16 *x, unsigned *bins, int N)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N; i += stride) {
        unsigned b = (*(unsigned short*)&x[i] >> 7) & 0xFF;  // 8 bits of exp
        atomicAdd(&bins[b], 1u);
    }
}

extern "C" __launch_bounds__(256, 4) __global__ void hist_v2_smem(
    const __nv_bfloat16 *x, unsigned *bins, int N)
{
    __shared__ unsigned smem_bins[256];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int t = threadIdx.x;
    if (t < 256) smem_bins[t] = 0;
    __syncthreads();

    int stride = gridDim.x * blockDim.x;
    // 8-ILP coalesced loads: uint4 = 16 B = 8 BF16
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

// v3: spread bins across 32 SMEM copies to avoid bank conflicts
extern "C" __launch_bounds__(256, 4) __global__ void hist_v3_spread(
    const __nv_bfloat16 *x, unsigned *bins, int N)
{
    // 256 bins × 32 copies = 8192 entries = 32 KB per block
    __shared__ unsigned smem_bins[256 * 32];
    int t = threadIdx.x;
    int lane = t & 31;
    // zero shmem
    #pragma unroll
    for (int j = t; j < 256 * 32; j += blockDim.x) smem_bins[j] = 0;
    __syncthreads();

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int warp_base = (blockIdx.x * blockDim.x + (threadIdx.x & ~31)) * 8;
    for (int i = warp_base + lane * 8; i < N - 7; i += stride * 8) {
        uint4 v = *(uint4*)&x[i];
        const unsigned short *p = (const unsigned short*)&v;
        #pragma unroll
        for (int j = 0; j < 8; j++) {
            unsigned b = (p[j] >> 7) & 0xFF;
            atomicAdd_block(&smem_bins[b * 32 + lane], 1u);
        }
    }
    __syncthreads();
    // Reduce 32 copies into 1 then store global
    if (t < 256) {
        unsigned sum = 0;
        #pragma unroll
        for (int k = 0; k < 32; k++) sum += smem_bins[t * 32 + k];
        atomicAdd(&bins[t], sum);
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = 1ull * 1024 * 1024 * 1024;
    int N = bytes / sizeof(__nv_bfloat16);

    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes);
    srand(42);
    for (int i = 0; i < N; i++) h[i] = __float2bfloat16(((float)rand() / RAND_MAX) * 2 - 1);

    __nv_bfloat16 *d_x; cudaMalloc(&d_x, bytes);
    cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice);
    free(h);

    unsigned *d_bins; cudaMalloc(&d_bins, 256 * sizeof(unsigned));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = 148 * 4, threads = 256;

    auto bench = [&](auto launch, const char* label) {
        cudaMemset(d_bins, 0, 256 * sizeof(unsigned));
        for (int i = 0; i < 3; i++) { cudaMemset(d_bins, 0, 256 * sizeof(unsigned)); launch(); }
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaMemset(d_bins, 0, 256 * sizeof(unsigned));
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        unsigned h_bins[256]; cudaMemcpy(h_bins, d_bins, 256*4, cudaMemcpyDeviceToHost);
        unsigned long total = 0; unsigned peak_bin = 0; int peak_idx = 0;
        for (int i = 0; i < 256; i++) {
            total += h_bins[i];
            if (h_bins[i] > peak_bin) { peak_bin = h_bins[i]; peak_idx = i; }
        }
        double gbs = bytes / (best/1000) / 1e9;
        printf("  %s: %.4f ms = %.1f GB/s (%.1f%% of 7300 HBM peak); total=%lu, peak bin %d=%u\n",
               label, best, gbs, gbs/7300*100, total, peak_idx, peak_bin);
    };

    bench([&]{ hist_v1_naive<<<blocks, threads>>>(d_x, d_bins, N); }, "v1 naive global atomics ");
    bench([&]{ hist_v2_smem<<<blocks, threads>>>(d_x, d_bins, N); }, "v2 smem aggregation     ");
    bench([&]{ hist_v3_spread<<<blocks, threads>>>(d_x, d_bins, N); }, "v3 smem 32-way spread   ");

    return 0;
}
