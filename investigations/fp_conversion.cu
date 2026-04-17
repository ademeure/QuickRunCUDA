// FP32 ↔ FP16/BF16 conversion throughput
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

__global__ void f32_to_f16(__half *out, float *in, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N; i += stride) out[i] = __float2half(in[i]);
    }
}

__global__ void f16_to_f32(float *out, __half *in, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N; i += stride) out[i] = __half2float(in[i]);
    }
}

__global__ void f32_to_bf16(__nv_bfloat16 *out, float *in, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N; i += stride) out[i] = __float2bfloat16(in[i]);
    }
}

__global__ void f32_to_f16x2(__half2 *out, float2 *in, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int it = 0; it < iters; it++) {
        for (int i = tid; i < N; i += stride) out[i] = __float22half2_rn(in[i]);
    }
}

int main() {
    cudaSetDevice(0);
    int N = 1024 * 1024 * 16;  // 16M elements

    float *d_f32; cudaMalloc(&d_f32, N * sizeof(float));
    __half *d_f16; cudaMalloc(&d_f16, N * sizeof(__half));
    __nv_bfloat16 *d_bf16; cudaMalloc(&d_bf16, N * sizeof(__nv_bfloat16));
    cudaMemset(d_f32, 0, N * sizeof(float));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int iters = 50;

    auto bench = [&](auto launch, const char *name) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 3; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total = (long)N * iters;
        double gops = total / (best/1000.0) / 1e9;
        printf("  %-30s %.3f ms = %.0f Gconv/s\n", name, best, gops);
    };

    printf("# B300 FP conversion throughput (16M elem × 50 iter, mostly mem-bound)\n\n");
    bench([&]{ f32_to_f16<<<148, 256>>>(d_f16, d_f32, N, iters); }, "F32 → F16 scalar");
    bench([&]{ f32_to_bf16<<<148, 256>>>(d_bf16, d_f32, N, iters); }, "F32 → BF16 scalar");
    bench([&]{ f16_to_f32<<<148, 256>>>(d_f32, d_f16, N, iters); }, "F16 → F32 scalar");
    bench([&]{ f32_to_f16x2<<<148, 256>>>((__half2*)d_f16, (float2*)d_f32, N/2, iters); },
          "F32 → F16 packed (x2)");

    return 0;
}
