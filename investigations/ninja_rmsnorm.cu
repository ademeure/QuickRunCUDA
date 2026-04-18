// NINJA RMS norm + GeLU + bias fused — should be HBM-bound
// Theoretical: 1 read input + 1 read weight + 1 read bias + 1 write output
// = 4 bytes per BF16 elem (= 2 R + 2 W) → at 6.7 TB/s mixed ceiling
// For 1M tokens × 4096 dim × 2 B = 8 GB elements, total HBM = 16 GB

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define DIM 4096
#define EPS 1e-6f

// One block per row. RMS norm: y = x * g / sqrt(mean(x^2) + eps), then GeLU(y + bias)
__launch_bounds__(256, 4) __global__ void rms_gelu_bias(
    const __nv_bfloat16 *x,
    const __nv_bfloat16 *g,    // weight (DIM,)
    const __nv_bfloat16 *bias, // bias (DIM,)
    __nv_bfloat16 *y, int n_rows)
{
    int row = blockIdx.x;
    if (row >= n_rows) return;
    int t = threadIdx.x;
    int per_thread = DIM / 256;  // 4096/256 = 16

    __shared__ float reduce_smem[16];

    // Pass 1: load row + compute sum-of-squares; KEEP IN REGISTERS
    int my_off = t * per_thread;
    float sum_sq = 0.f;
    uint4 v0 = *(uint4*)&x[row * DIM + my_off];
    uint4 v1 = *(uint4*)&x[row * DIM + my_off + 8];
    const __nv_bfloat16 *p0 = (const __nv_bfloat16*)&v0;
    const __nv_bfloat16 *p1 = (const __nv_bfloat16*)&v1;
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float v = __bfloat162float(p0[i]); sum_sq += v * v;
        v = __bfloat162float(p1[i]); sum_sq += v * v;
    }

    // Block reduce sum_sq
    int lane = t & 31, warp = t >> 5;
    for (int o = 16; o; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
    if (lane == 0) reduce_smem[warp] = sum_sq;
    __syncthreads();
    if (warp == 0) {
        sum_sq = (lane < 8) ? reduce_smem[lane] : 0.f;
        for (int o = 4; o; o >>= 1) sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, o);
        if (lane == 0) reduce_smem[0] = sum_sq;
    }
    __syncthreads();
    float rsqrt_val = rsqrtf(reduce_smem[0] / DIM + EPS);

    // Pass 2: load g and bias for THIS thread's portion + compute fused output
    uint4 g0 = *(uint4*)&g[my_off];
    uint4 g1 = *(uint4*)&g[my_off + 8];
    uint4 b0 = *(uint4*)&bias[my_off];
    uint4 b1 = *(uint4*)&bias[my_off + 8];
    const __nv_bfloat16 *gp0 = (const __nv_bfloat16*)&g0;
    const __nv_bfloat16 *gp1 = (const __nv_bfloat16*)&g1;
    const __nv_bfloat16 *bp0 = (const __nv_bfloat16*)&b0;
    const __nv_bfloat16 *bp1 = (const __nv_bfloat16*)&b1;

    __nv_bfloat16 out0[8], out1[8];
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        // RMS: scaled = x * g * rsqrt_val
        float v = __bfloat162float(p0[i]) * __bfloat162float(gp0[i]) * rsqrt_val;
        // Add bias
        v += __bfloat162float(bp0[i]);
        // Sigmoid GeLU: x * sigmoid(1.702*x) = x / (1 + exp(-1.702*x))
        float s = 1.0f / (1.0f + __expf(-1.702f * v));
        out0[i] = __float2bfloat16(v * s);
    }
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        float v = __bfloat162float(p1[i]) * __bfloat162float(gp1[i]) * rsqrt_val;
        v += __bfloat162float(bp1[i]);
        float s = 1.0f / (1.0f + __expf(-1.702f * v));
        out1[i] = __float2bfloat16(v * s);
    }
    *(uint4*)&y[row * DIM + my_off] = *(uint4*)out0;
    *(uint4*)&y[row * DIM + my_off + 8] = *(uint4*)out1;
}

int main() {
    cudaSetDevice(0);
    int n_rows = 256 * 1024;  // 256K tokens
    size_t bytes_x = (size_t)n_rows * DIM * sizeof(__nv_bfloat16);  // 2 GB
    size_t bytes_w = DIM * sizeof(__nv_bfloat16);  // 8 KB

    __nv_bfloat16 *d_x, *d_y, *d_g, *d_bias;
    cudaMalloc(&d_x, bytes_x); cudaMalloc(&d_y, bytes_x);
    cudaMalloc(&d_g, bytes_w); cudaMalloc(&d_bias, bytes_w);

    __nv_bfloat16 *h = (__nv_bfloat16*)malloc(bytes_x);
    srand(42);
    for (size_t i = 0; i < (size_t)n_rows*DIM; i++)
        h[i] = __float2bfloat16(((float)rand()/RAND_MAX)*2-1);
    cudaMemcpy(d_x, h, bytes_x, cudaMemcpyHostToDevice);
    free(h);

    __nv_bfloat16 *hw = (__nv_bfloat16*)malloc(bytes_w);
    for (int i = 0; i < DIM; i++) hw[i] = __float2bfloat16(1.0f);
    cudaMemcpy(d_g, hw, bytes_w, cudaMemcpyHostToDevice);
    for (int i = 0; i < DIM; i++) hw[i] = __float2bfloat16(0.01f);
    cudaMemcpy(d_bias, hw, bytes_w, cudaMemcpyHostToDevice);
    free(hw);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 5; i++)
        rms_gelu_bias<<<n_rows, 256>>>(d_x, d_g, d_bias, d_y, n_rows);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        rms_gelu_bias<<<n_rows, 256>>>(d_x, d_g, d_bias, d_y, n_rows);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    // HBM traffic: 1× R(x) + 1× W(y) per row + small g, bias (cached after warm)
    size_t actual = bytes_x * 2;
    double gbs = actual / (best/1000) / 1e9;
    double sol_us = (double)bytes_x * 2 / 7300e9 * 1e6;
    printf("# RMS+GeLU+bias fused, %d rows × %d dim BF16:\n", n_rows, DIM);
    printf("  Best: %.4f ms = %.0f GB/s = %.1f%% HBM peak\n", best, gbs, gbs/7300*100);
    printf("  SoL bound (1R+1W): %.1f us; ratio %.2fx\n", sol_us, best*1000/sol_us);
    return 0;
}
