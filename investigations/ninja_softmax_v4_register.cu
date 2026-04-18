// NINJA softmax: multi-row per block to amortize syncthreads overhead
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#ifndef ROWS_PER_BLK
#define ROWS_PER_BLK 4
#endif

extern "C" __launch_bounds__(256, 4) __global__ void softmax_multi_row(
    const __nv_bfloat16 *x, __nv_bfloat16 *y, int row_len)
{
    int row = blockIdx.x * ROWS_PER_BLK;
    int t = threadIdx.x;
    int per_thread = row_len / 256;  // 4096/256 = 16
    int my_off = t * per_thread;

    extern __shared__ float reduce_smem[];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_BLK; r++) {
        const __nv_bfloat16 *row_x = x + (row + r) * row_len;
        __nv_bfloat16 *row_y = y + (row + r) * row_len;

        // Pass 1: max
        float m = -1e30f;
        // 16 elements = 2 uint4
        uint4 v0 = *(uint4*)&row_x[my_off];
        uint4 v1 = *(uint4*)&row_x[my_off + 8];
        const __nv_bfloat16 *p0 = (const __nv_bfloat16*)&v0;
        const __nv_bfloat16 *p1 = (const __nv_bfloat16*)&v1;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            m = fmaxf(m, __bfloat162float(p0[i]));
            m = fmaxf(m, __bfloat162float(p1[i]));
        }
        for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        int warp = t >> 5, lane = t & 31;
        if (lane == 0) reduce_smem[warp] = m;
        __syncthreads();
        if (warp == 0) {
            m = (lane < 8) ? reduce_smem[lane] : -1e30f;
            for (int o = 4; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
            if (lane == 0) reduce_smem[0] = m;
        }
        __syncthreads();
        float row_max = reduce_smem[0];

        // Pass 2: sum
        float s = 0.f;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            s += __expf(__bfloat162float(p0[i]) - row_max);
            s += __expf(__bfloat162float(p1[i]) - row_max);
        }
        for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) reduce_smem[warp + 8] = s;
        __syncthreads();
        if (warp == 0) {
            s = (lane < 8) ? reduce_smem[lane + 8] : 0.f;
            for (int o = 4; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
            if (lane == 0) reduce_smem[1] = s;
        }
        __syncthreads();
        float inv_sum = 1.0f / reduce_smem[1];

        // Pass 3: write
        __nv_bfloat16 buf0[8], buf1[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            buf0[i] = __float2bfloat16(__expf(__bfloat162float(p0[i]) - row_max) * inv_sum);
            buf1[i] = __float2bfloat16(__expf(__bfloat162float(p1[i]) - row_max) * inv_sum);
        }
        *(uint4*)&row_y[my_off] = *(uint4*)buf0;
        *(uint4*)&row_y[my_off + 8] = *(uint4*)buf1;
    }
}

int main() {
    cudaSetDevice(0);
    int row_len = 4096;
    int n_rows = 256 * 1024;
    size_t bytes = (size_t)n_rows * row_len * sizeof(__nv_bfloat16);

    __nv_bfloat16 *d_x, *d_y;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes);
    srand(42);
    for (size_t i = 0; i < (size_t)n_rows*row_len; i++)
        h[i] = __float2bfloat16(((float)rand()/RAND_MAX)*4-2);
    cudaMemcpy(d_x, h, bytes, cudaMemcpyHostToDevice);
    free(h);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = n_rows / ROWS_PER_BLK;

    for (int i = 0; i < 3; i++)
        softmax_multi_row<<<blocks, 256, 16*sizeof(float)>>>(d_x, d_y, row_len);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        softmax_multi_row<<<blocks, 256, 16*sizeof(float)>>>(d_x, d_y, row_len);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    size_t actual = bytes * 2;
    double a_gbs = actual / (best/1000) / 1e9;
    printf("# ROWS_PER_BLK=%d: %.4f ms = %.0f GB/s = %.1f%% HBM\n",
           ROWS_PER_BLK, best, a_gbs, a_gbs/7300*100);
    return 0;
}
