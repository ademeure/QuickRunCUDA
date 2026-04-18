// A2: RMS norm + GeLU + bias fused — what's the BF16 SoL?
//
// THEORETICAL:
//   Per element: read input (2B BF16) + write output (2B BF16) = 4 B / element
//   Read weight (D BF16): broadcast, amortized one-time per row
//   Read bias  (D BF16):  broadcast, amortized one-time per row
//   For N rows × D=4096, weight+bias = 16 KB — fits in L1 easily.
//
//   HBM3E peak = 7.31 TB/s
//   SoL throughput = 7.31 / 4 = 1.83 G elements/s for full-fused
//
// FORMULA:
//   y[i,j] = bias[j] + weight[j] * x[i,j] / sqrt(mean(x[i,k]^2 for k) + eps) × GeLU
//   GeLU(z) = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))
//
// Or: y = gelu(rms_norm(x) * w) + b
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cmath>

constexpr int D = 4096;
constexpr int N_ROWS = 1024 * 1024;  // 1M tokens

// Naive non-fused kernels for comparison

__launch_bounds__(256, 4) __global__ void k_rms(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                                 __nv_bfloat16 *y, float eps) {
    int row = blockIdx.x;
    extern __shared__ float smem[];
    // Compute sum of squares
    float local_sum = 0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float v = __bfloat162float(x[row * D + j]);
        local_sum += v * v;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(smem[0] / D + eps);
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float v = __bfloat162float(x[row * D + j]) * inv_norm * __bfloat162float(w[j]);
        y[row * D + j] = __float2bfloat16(v);
    }
}

__launch_bounds__(256, 8) __global__ void k_gelu(__nv_bfloat16 *y, const __nv_bfloat16 *bias) {
    long N = (long)N_ROWS * D;
    long tid = (long)blockIdx.x * blockDim.x + threadIdx.x;
    long stride = (long)gridDim.x * blockDim.x;
    for (long i = tid; i < N; i += stride) {
        float v = __bfloat162float(y[i]);
        float b = __bfloat162float(bias[i % D]);
        float z = v + b;
        // Approx GeLU: 0.5*z*(1+tanh(sqrt(2/pi)*(z+0.044715*z^3)))
        float g = 0.5f * z * (1.0f + tanhf(0.7978845f * (z + 0.044715f * z * z * z)));
        y[i] = __float2bfloat16(g);
    }
}

// Fused kernel: 1 row per block, all 3 ops in single pass
__launch_bounds__(256, 4) __global__ void k_fused(const __nv_bfloat16 *x, const __nv_bfloat16 *w,
                                                   const __nv_bfloat16 *bias, __nv_bfloat16 *y, float eps) {
    int row = blockIdx.x;
    extern __shared__ float smem[];
    // Load row into registers via vectorized BF16 reads (8 bf16 = 16 B per thread per chunk)
    // Use pre-pass to compute sum of squares
    float local_sum = 0;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float v = __bfloat162float(x[row * D + j]);
        local_sum += v * v;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(smem[0] / D + eps);
    // Final pass: norm + scale + bias + GeLU
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float xv = __bfloat162float(x[row * D + j]);
        float wv = __bfloat162float(w[j]);
        float bv = __bfloat162float(bias[j]);
        float normed = xv * inv_norm * wv + bv;
        float g = 0.5f * normed * (1.0f + tanhf(0.7978845f * (normed + 0.044715f * normed * normed * normed)));
        y[row * D + j] = __float2bfloat16(g);
    }
}

// Vectorized fused: BF162 (8 BF16 per uint128)
__launch_bounds__(256, 4) __global__ void k_fused_vec(const __nv_bfloat162 *x2,
                                                       const __nv_bfloat162 *w2,
                                                       const __nv_bfloat162 *bias2,
                                                       __nv_bfloat162 *y2, float eps) {
    int row = blockIdx.x;
    extern __shared__ float smem[];
    constexpr int D2 = D / 2;  // 2048
    float local_sum = 0;
    for (int j = threadIdx.x; j < D2; j += blockDim.x) {
        __nv_bfloat162 v2 = x2[row * D2 + j];
        float v0 = __bfloat162float(__low2bfloat16(v2));
        float v1 = __bfloat162float(__high2bfloat16(v2));
        local_sum += v0 * v0 + v1 * v1;
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    float inv_norm = rsqrtf(smem[0] / D + eps);
    for (int j = threadIdx.x; j < D2; j += blockDim.x) {
        __nv_bfloat162 xv2 = x2[row * D2 + j];
        __nv_bfloat162 wv2 = w2[j];
        __nv_bfloat162 bv2 = bias2[j];
        float xa = __bfloat162float(__low2bfloat16(xv2));
        float xb = __bfloat162float(__high2bfloat16(xv2));
        float wa = __bfloat162float(__low2bfloat16(wv2));
        float wb = __bfloat162float(__high2bfloat16(wv2));
        float ba = __bfloat162float(__low2bfloat16(bv2));
        float bb = __bfloat162float(__high2bfloat16(bv2));
        float na = xa * inv_norm * wa + ba;
        float nb = xb * inv_norm * wb + bb;
        float ga = 0.5f * na * (1.0f + tanhf(0.7978845f * (na + 0.044715f * na * na * na)));
        float gb = 0.5f * nb * (1.0f + tanhf(0.7978845f * (nb + 0.044715f * nb * nb * nb)));
        y2[row * D2 + j] = __nv_bfloat162{__float2bfloat16(ga), __float2bfloat16(gb)};
    }
}

int main() {
    cudaSetDevice(0);
    size_t bytes = (size_t)N_ROWS * D * 2;  // BF16
    __nv_bfloat16 *d_x, *d_y, *d_w, *d_bias;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_w, D * 2);
    cudaMalloc(&d_bias, D * 2);
    cudaMemset(d_x, 0x3c, bytes);
    cudaMemset(d_w, 0x3f, D * 2);
    cudaMemset(d_bias, 0x3c, D * 2);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        // Bytes: 1 read input + 1 write output = 2*bytes
        // (weight + bias amortize to 0)
        double tbs = (double)(2 * bytes) / (best/1000.0) / 1e12;
        double pct = tbs / 7.31 * 100;
        double elements_per_s = (double)N_ROWS * D / (best/1000.0);
        printf("  %-30s  %.3f ms  %.2f TB/s (%.1f%% HBM)  %.2f G elem/s\n",
               name, best, tbs, pct, elements_per_s/1e9);
    };

    int blocks_per_row = N_ROWS;  // 1 block per row
    int threads = 256;
    int smem = threads * sizeof(float);

    bench("RMS-only (1 row/block)", [&]() {
        k_rms<<<blocks_per_row, threads, smem>>>(d_x, d_w, d_y, 1e-5f);
    });
    bench("RMS + GeLU separate", [&]() {
        k_rms<<<blocks_per_row, threads, smem>>>(d_x, d_w, d_y, 1e-5f);
        k_gelu<<<148*8, 256>>>(d_y, d_bias);
    });
    bench("Fused RMS+GeLU+bias (scalar)", [&]() {
        k_fused<<<blocks_per_row, threads, smem>>>(d_x, d_w, d_bias, d_y, 1e-5f);
    });
    bench("Fused vectorized (BF162)", [&]() {
        k_fused_vec<<<blocks_per_row, threads, smem>>>(
            (const __nv_bfloat162*)d_x, (const __nv_bfloat162*)d_w,
            (const __nv_bfloat162*)d_bias, (__nv_bfloat162*)d_y, 1e-5f);
    });

    return 0;
}
