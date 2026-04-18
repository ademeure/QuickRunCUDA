// Free rein: online (Welford-style) softmax — 2-pass instead of 3-pass.
// Should hit ~90% of HBM peak vs L4's 70% (3-pass).
//
// Pass 1: compute (max, sum) via online algorithm in 1 read sweep
// Pass 2: read again + write normalized output

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" __launch_bounds__(256, 4) __global__ void softmax_online(
    const __nv_bfloat16 *x, __nv_bfloat16 *y, int row_len)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    const __nv_bfloat16 *row_x = x + row * row_len;
    __nv_bfloat16 *row_y = y + row * row_len;

    extern __shared__ float smem[];  // [0..15] for max reduce, [16..31] for sum

    int per_thread = row_len / 256;

    // Pass 1: ONLINE max + sum in one sweep
    float m = -1e30f, s = 0.f;
    for (int i = 0; i < per_thread; i++) {
        float v = __bfloat162float(row_x[i * 256 + t]);
        float new_m = fmaxf(m, v);
        s = s * __expf(m - new_m) + __expf(v - new_m);
        m = new_m;
    }

    // Block-level online reduce: warp-shfl combining max+sum
    int warp = t >> 5;
    int lane = t & 31;
    for (int o = 16; o; o >>= 1) {
        float other_m = __shfl_xor_sync(0xffffffff, m, o);
        float other_s = __shfl_xor_sync(0xffffffff, s, o);
        float new_m = fmaxf(m, other_m);
        s = s * __expf(m - new_m) + other_s * __expf(other_m - new_m);
        m = new_m;
    }

    if (lane == 0) {
        smem[warp] = m;
        smem[warp + 16] = s;
    }
    __syncthreads();

    if (warp == 0) {
        m = (lane < blockDim.x / 32) ? smem[lane] : -1e30f;
        s = (lane < blockDim.x / 32) ? smem[lane + 16] : 0.f;
        for (int o = 4; o; o >>= 1) {
            float other_m = __shfl_xor_sync(0xffffffff, m, o);
            float other_s = __shfl_xor_sync(0xffffffff, s, o);
            float new_m = fmaxf(m, other_m);
            s = s * __expf(m - new_m) + other_s * __expf(other_m - new_m);
            m = new_m;
        }
        if (lane == 0) {
            smem[0] = m;
            smem[16] = s;
        }
    }
    __syncthreads();

    float row_max = smem[0];
    float inv_sum = 1.0f / smem[16];

    // Pass 2: read again + write normalized
    for (int i = 0; i < per_thread; i++) {
        float v = __bfloat162float(row_x[i * 256 + t]);
        row_y[i * 256 + t] = __float2bfloat16(__expf(v - row_max) * inv_sum);
    }
}

int main() {
    cudaSetDevice(0);
    int row_len = 4096;
    int n_rows = 256 * 1024;
    size_t bytes = (size_t)n_rows * row_len * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_x; cudaMalloc(&d_x, bytes);
    __nv_bfloat16 *d_y; cudaMalloc(&d_y, bytes);

    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < (size_t)n_rows * row_len; i++)
        h[i] = __float2bfloat16(((float)rand()/RAND_MAX) * 4 - 2);
    cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice);
    free(h);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int threads = 256;
    int smem_bytes = 32 * sizeof(float);

    for (int i = 0; i < 3; i++)
        softmax_online<<<n_rows, threads, smem_bytes>>>(d_x, d_y, row_len);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        softmax_online<<<n_rows, threads, smem_bytes>>>(d_x, d_y, row_len);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    size_t actual_bytes = bytes * 3;  // 2× R + 1× W
    size_t effective_bytes = bytes * 2;  // 1× R + 1× W (SoL bound)
    double actual_gbs = actual_bytes / (best/1000) / 1e9;
    double effective_gbs = effective_bytes / (best/1000) / 1e9;
    double sol_us = (double)bytes * 2 / 7300e9 * 1e6;

    printf("# softmax_online (2-pass): %.4f ms\n", best);
    printf("  Actual traffic (2R+1W): %.0f GB/s = %.1f%% of HBM peak\n",
           actual_gbs, actual_gbs/7300*100);
    printf("  Effective (R+W once)  : %.0f GB/s = %.1f%% of HBM peak\n",
           effective_gbs, effective_gbs/7300*100);
    printf("  SoL bound: %.1f us; ratio %.2fx slower\n",
           sol_us, best * 1000 / sol_us);
    printf("  vs L4 3-pass (5106 GB/s actual, 1.26 ms): %.2fx speedup\n",
           1.26 / best);
    return 0;
}
