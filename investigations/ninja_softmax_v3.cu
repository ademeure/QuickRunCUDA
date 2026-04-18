// NINJA softmax v3: cache row in SHMEM so HBM traffic is just 1R + 1W
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

extern "C" __launch_bounds__(512, 4) __global__ void softmax_smem(
    const __nv_bfloat16 *x, __nv_bfloat16 *y, int row_len)
{
    int row = blockIdx.x;
    int t = threadIdx.x;
    const __nv_bfloat16 *row_x = x + row * row_len;
    __nv_bfloat16 *row_y = y + row * row_len;

    extern __shared__ char smem_raw[];
    __nv_bfloat16 *row_smem = (__nv_bfloat16*)smem_raw;
    float *reduce_smem = (float*)(smem_raw + row_len * sizeof(__nv_bfloat16));

    int per_thread = row_len / 512;
    int my_off = t * per_thread;

    // Pass 1: load row to SMEM + per-thread max
    float m = -1e30f;
    if (per_thread == 8) {
        uint4 v = *(uint4*)&row_x[my_off];
        *(uint4*)&row_smem[my_off] = v;  // cache to SMEM
        const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
        #pragma unroll
        for (int i = 0; i < 8; i++) m = fmaxf(m, __bfloat162float(p[i]));
    }

    int warp = t >> 5, lane = t & 31;
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    if (lane == 0) reduce_smem[warp] = m;
    __syncthreads();
    if (warp == 0) {
        m = (lane < 16) ? reduce_smem[lane] : -1e30f;
        for (int o = 8; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
        if (lane == 0) reduce_smem[0] = m;
    }
    __syncthreads();
    float row_max = reduce_smem[0];

    // Pass 2: sum of exp from SMEM (no HBM read!)
    float s = 0.f;
    {
        uint4 v = *(uint4*)&row_smem[my_off];
        const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
        #pragma unroll
        for (int i = 0; i < 8; i++) s += __expf(__bfloat162float(p[i]) - row_max);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    if (lane == 0) reduce_smem[warp + 16] = s;
    __syncthreads();
    if (warp == 0) {
        s = (lane < 16) ? reduce_smem[lane + 16] : 0.f;
        for (int o = 8; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
        if (lane == 0) reduce_smem[1] = s;
    }
    __syncthreads();
    float inv_sum = 1.0f / reduce_smem[1];

    // Pass 3: write normalized from SMEM (no HBM read; only write to HBM)
    {
        uint4 v = *(uint4*)&row_smem[my_off];
        const __nv_bfloat16 *p = (const __nv_bfloat16*)&v;
        __nv_bfloat16 outbuf[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            outbuf[i] = __float2bfloat16(__expf(__bfloat162float(p[i]) - row_max) * inv_sum);
        }
        *(uint4*)&row_y[my_off] = *(uint4*)outbuf;
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
    int smem_bytes = row_len * sizeof(__nv_bfloat16) + 64 * sizeof(float);

    // Need to allow large dynamic SMEM
    cudaFuncSetAttribute(softmax_smem, cudaFuncAttributeMaxDynamicSharedMemorySize, 16384);

    for (int i = 0; i < 3; i++)
        softmax_smem<<<n_rows, 512, smem_bytes>>>(d_x, d_y, row_len);
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(err)); return 1; }

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        softmax_smem<<<n_rows, 512, smem_bytes>>>(d_x, d_y, row_len);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    size_t actual = bytes * 2;  // 1R + 1W only (rest from SMEM)
    double a_gbs = actual / (best/1000) / 1e9;
    double sol = (double)bytes*2/7300e9*1e6;
    printf("# softmax_smem: %.4f ms\n", best);
    printf("  HBM traffic (1R+1W): %.0f GB/s = %.1f%% of HBM peak\n",
           a_gbs, a_gbs/7300*100);
    printf("  SoL bound: %.1f us; ratio %.2fx\n", sol, best*1000/sol);

    return 0;
}
