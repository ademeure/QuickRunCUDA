// A2 optimized: 1 warp per row + REDUX.SUM, vectorized BF16x8 LDG.128
//
// 256 threads/block × 8 warps × N_blocks rows → 8 rows per block
// Each warp: 32 threads × 128 BF16 elements per thread (= 4096 D)
// Use ldg.128 = uint4 → 8 BF16 per load (16 B vector)
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

// 1 warp per row, vectorized x8 BF16 loads
__launch_bounds__(256, 2) __global__ void k_warp_per_row(
        const uint4 *x, const uint4 *w, const uint4 *bias, uint4 *y, float eps) {
    int warp_id = threadIdx.x >> 5;
    int lane = threadIdx.x & 31;
    long row = (long)blockIdx.x * 8 + warp_id;
    if (row >= N_ROWS) return;
    constexpr int D8 = D / 8;  // 512 (uint4 of 8 BF16)
    // Each thread does 16 LDG.128 to cover the row (32 lanes × 16 = 512)
    constexpr int PER_THREAD = D8 / 32;  // 16

    // Pass 1: sum of squares
    uint4 row_data[PER_THREAD];
    float sumsq = 0;
    #pragma unroll
    for (int k = 0; k < PER_THREAD; k++) {
        uint4 v = x[row * D8 + k * 32 + lane];
        row_data[k] = v;
        // Unpack 8 BF16
        __nv_bfloat162 *bf2 = (__nv_bfloat162*)&v;
        for (int i = 0; i < 4; i++) {
            float a = __bfloat162float(__low2bfloat16(bf2[i]));
            float b = __bfloat162float(__high2bfloat16(bf2[i]));
            sumsq += a * a + b * b;
        }
    }
    float total = warp_sum(sumsq);
    float inv_norm = rsqrtf(total / D + eps);

    // Pass 2: norm + scale + bias + GeLU + write
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
            float xb = __bfloat162float(__high2bfloat16(xb2[i]));
            float wa = __bfloat162float(__low2bfloat16(wb2[i]));
            float wb = __bfloat162float(__high2bfloat16(wb2[i]));
            float ba = __bfloat162float(__low2bfloat16(bb2[i]));
            float bb_ = __bfloat162float(__high2bfloat16(bb2[i]));
            float na = xa * inv_norm * wa + ba;
            float nb = xb * inv_norm * wb + bb_;
            // Fast GeLU: x * sigmoid(1.702*x), sigmoid via 1/(1+exp(-y))
            // sigmoid(y) = 1/(1+exp(-y)) ≈ 0.5*(1 + tanh(y/2)) — but tanh slow.
            // Use approximation via 1/(1+exp): use __frcp_rn(1+__expf(-y))
            float ya = 1.702f * na, yb = 1.702f * nb;
            float ga = na * __frcp_rn(1.0f + __expf(-ya));
            float gb = nb * __frcp_rn(1.0f + __expf(-yb));
            ob2[i] = __nv_bfloat162{__float2bfloat16(ga), __float2bfloat16(gb)};
        }
        y[row * D8 + k * 32 + lane] = out;
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = (size_t)N_ROWS * D * 2;
    void *d_x, *d_y; cudaMalloc(&d_x, bytes); cudaMalloc(&d_y, bytes);
    void *d_w, *d_bias; cudaMalloc(&d_w, D * 2); cudaMalloc(&d_bias, D * 2);
    cudaMemset(d_x, 0x3c, bytes);
    cudaMemset(d_w, 0x3f, D * 2);
    cudaMemset(d_bias, 0x3c, D * 2);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = (N_ROWS + 7) / 8;
    auto launch = [&]() {
        k_warp_per_row<<<blocks, 256>>>((uint4*)d_x, (uint4*)d_w, (uint4*)d_bias, (uint4*)d_y, 1e-5f);
    };
    for (int i = 0; i < 3; i++) launch();
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    float best = 1e30f;
    for (int i = 0; i < 8; i++) {
        cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double tbs = (double)(2 * bytes) / (best/1000.0) / 1e12;
    printf("# Warp-per-row vectorized fused RMS+GeLU+bias\n");
    printf("  best=%.3f ms  %.2f TB/s (%.1f%% of 7.31 TB/s HBM)\n",
           best, tbs, tbs/7.31*100);
    return 0;
}
