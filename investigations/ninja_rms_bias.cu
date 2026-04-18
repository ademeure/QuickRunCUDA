// RMS+bias only (no GeLU) — to isolate compute cost of GeLU
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
constexpr int D = 4096;
constexpr int N_ROWS = 1024 * 1024;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int s = 16; s > 0; s >>= 1) v += __shfl_xor_sync(0xFFFFFFFF, v, s);
    return v;
}

__launch_bounds__(256, 2) __global__ void k_rms_bias(
        const uint4 *x, const uint4 *w, const uint4 *bias, uint4 *y, float eps) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    long row = (long)blockIdx.x * 8 + warp_id;
    if (row >= N_ROWS) return;
    constexpr int D8 = D / 8;
    constexpr int PER_THREAD = D8 / 32;
    uint4 row_data[PER_THREAD];
    float sumsq = 0;
    #pragma unroll
    for (int k = 0; k < PER_THREAD; k++) {
        uint4 v = x[row * D8 + k * 32 + lane];
        row_data[k] = v;
        __nv_bfloat162 *bf2 = (__nv_bfloat162*)&v;
        for (int i = 0; i < 4; i++) {
            float a = __bfloat162float(__low2bfloat16(bf2[i]));
            float b = __bfloat162float(__high2bfloat16(bf2[i]));
            sumsq += a * a + b * b;
        }
    }
    float total = warp_sum(sumsq);
    float inv_norm = rsqrtf(total / D + eps);
    #pragma unroll
    for (int k = 0; k < PER_THREAD; k++) {
        uint4 wv = w[k * 32 + lane];
        uint4 bv = bias[k * 32 + lane];
        __nv_bfloat162 *xb2 = (__nv_bfloat162*)&row_data[k];
        __nv_bfloat162 *wb2 = (__nv_bfloat162*)&wv;
        __nv_bfloat162 *bb2 = (__nv_bfloat162*)&bv;
        uint4 out;
        __nv_bfloat162 *ob2 = (__nv_bfloat162*)&out;
        for (int i = 0; i < 4; i++) {
            float xa = __bfloat162float(__low2bfloat16(xb2[i]));
            float xb_ = __bfloat162float(__high2bfloat16(xb2[i]));
            float wa = __bfloat162float(__low2bfloat16(wb2[i]));
            float wb_ = __bfloat162float(__high2bfloat16(wb2[i]));
            float ba = __bfloat162float(__low2bfloat16(bb2[i]));
            float bb_ = __bfloat162float(__high2bfloat16(bb2[i]));
            float na = xa * inv_norm * wa + ba;
            float nb = xb_ * inv_norm * wb_ + bb_;
            ob2[i] = __nv_bfloat162{__float2bfloat16(na), __float2bfloat16(nb)};
        }
        y[row * D8 + k * 32 + lane] = out;
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = (size_t)N_ROWS * D * 2;
    void *d_x, *d_y; cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    void *d_w, *d_bias; cudaMalloc(&d_w, D * 2); cudaMalloc(&d_bias, D * 2);
    cudaMemset(d_x, 0x3c, bytes); cudaMemset(d_w, 0x3f, D*2); cudaMemset(d_bias, 0x3c, D*2);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = (N_ROWS + 7) / 8;
    auto launch = [&]() {
        k_rms_bias<<<blocks, 256>>>((uint4*)d_x, (uint4*)d_w, (uint4*)d_bias, (uint4*)d_y, 1e-5f);
    };
    for (int i = 0; i < 3; i++) launch();
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 8; i++) {
        cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double tbs = (double)(2 * bytes) / (best/1000.0) / 1e12;
    printf("# RMS+bias only (NO GeLU): %.3f ms = %.2f TB/s (%.1f%% HBM)\n", best, tbs, tbs/7.31*100);
    return 0;
}
