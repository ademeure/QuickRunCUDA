// Simple 3-pass softmax for comparison
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

extern "C" __launch_bounds__(256, 4) __global__ void sm_max(const __nv_bfloat16 *x, float *m_out, int row_len) {
    int row = blockIdx.x; int t = threadIdx.x;
    const __nv_bfloat16 *r = x + row * row_len;
    int per_thread = row_len / 256;
    float m = -1e30f;
    for (int i = 0; i < per_thread; i++) m = fmaxf(m, __bfloat162float(r[i*256 + t]));
    for (int o = 16; o; o >>= 1) m = fmaxf(m, __shfl_xor_sync(0xffffffff, m, o));
    __shared__ float sm[8];
    if ((t&31)==0) sm[t>>5] = m;
    __syncthreads();
    if (t < 8) { float v = sm[t]; for (int o = 4; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xff, v, o)); if (t==0) m_out[row] = v; }
}
extern "C" __launch_bounds__(256, 4) __global__ void sm_expsum(const __nv_bfloat16 *x, __nv_bfloat16 *exp_out, const float *m_in, float *s_out, int row_len) {
    int row = blockIdx.x; int t = threadIdx.x;
    const __nv_bfloat16 *r = x + row * row_len;
    __nv_bfloat16 *o = exp_out + row * row_len;
    float m = m_in[row]; int per_thread = row_len / 256;
    float s = 0;
    for (int i = 0; i < per_thread; i++) {
        float e = __expf(__bfloat162float(r[i*256 + t]) - m);
        o[i*256 + t] = __float2bfloat16(e);
        s += e;
    }
    for (int o2 = 16; o2; o2 >>= 1) s += __shfl_xor_sync(0xffffffff, s, o2);
    __shared__ float sm[8];
    if ((t&31)==0) sm[t>>5] = s;
    __syncthreads();
    if (t < 8) { float v = sm[t]; for (int o2 = 4; o2; o2 >>= 1) v += __shfl_xor_sync(0xff, v, o2); if (t==0) s_out[row] = v; }
}
extern "C" __launch_bounds__(256, 4) __global__ void sm_div(const __nv_bfloat16 *exp_in, __nv_bfloat16 *y, const float *s_in, int row_len) {
    int row = blockIdx.x; int t = threadIdx.x;
    const __nv_bfloat16 *r = exp_in + row * row_len;
    __nv_bfloat16 *o = y + row * row_len;
    float inv_s = 1.0f / s_in[row];
    int per_thread = row_len / 256;
    for (int i = 0; i < per_thread; i++) o[i*256 + t] = __float2bfloat16(__bfloat162float(r[i*256 + t]) * inv_s);
}

int main() {
    cudaSetDevice(0);
    int row_len = 4096; int n_rows = 256 * 1024;
    size_t bytes = (size_t)n_rows * row_len * 2;
    __nv_bfloat16 *d_x, *d_y, *d_exp;
    cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes); cudaMalloc(&d_exp, bytes);
    float *d_m, *d_s; cudaMalloc(&d_m, n_rows*4); cudaMalloc(&d_s, n_rows*4);
    cudaMemset(d_x, 0x42, bytes);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto run = [&]() {
        sm_max<<<n_rows, 256>>>(d_x, d_m, row_len);
        sm_expsum<<<n_rows, 256>>>(d_x, d_exp, d_m, d_s, row_len);
        sm_div<<<n_rows, 256>>>(d_exp, d_y, d_s, row_len);
    };
    for (int i = 0; i < 3; i++) run();
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0); run(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    // Mem ops: max=R, exp=R+W, div=R+W; total 3R+2W = 5 ops
    double gbs = bytes * 5 / (best/1000.0) / 1e9;
    printf("3-pass softmax: %.4f ms = %.0f GB/s (5 mem ops/byte)\n", best, gbs);
    return 0;
}
