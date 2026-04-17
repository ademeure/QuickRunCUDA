// FP8 (E4M3, E5M2) conversion throughput
#include <cuda_runtime.h>
#include <cuda_fp8.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

__global__ void cvt_f32_to_e4m3(__nv_fp8_e4m3 *out, const float *in, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    __nv_fp8_e4m3 r0, r1, r2, r3;
    for (int i = 0; i < iters; i++) {
        for (int j = tid; j < N; j += stride) {
            float a = in[j];
            float b = in[j + 1];
            float c = in[j + 2];
            float d = in[j + 3];
            r0 = __nv_fp8_e4m3(a);
            r1 = __nv_fp8_e4m3(b);
            r2 = __nv_fp8_e4m3(c);
            r3 = __nv_fp8_e4m3(d);
            out[j] = r0;
            out[j+1] = r1;
            out[j+2] = r2;
            out[j+3] = r3;
        }
    }
}

__global__ void cvt_f32_to_e5m2(__nv_fp8_e5m2 *out, const float *in, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = 0; i < iters; i++) {
        for (int j = tid; j < N; j += stride) {
            out[j] = __nv_fp8_e5m2(in[j]);
        }
    }
}

__global__ void cvt_bf16_to_e4m3(__nv_fp8_e4m3 *out, const __nv_bfloat16 *in, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = 0; i < iters; i++) {
        for (int j = tid; j < N; j += stride) {
            out[j] = __nv_fp8_e4m3(__bfloat162float(in[j]));
        }
    }
}

__global__ void cvt_e4m3_to_f32(float *out, const __nv_fp8_e4m3 *in, int N, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = 0; i < iters; i++) {
        for (int j = tid; j < N; j += stride) {
            out[j] = float(in[j]);
        }
    }
}

int main() {
    cudaSetDevice(0);
    int N = 1024 * 1024;  // 1M elements

    float *d_f32; cudaMalloc(&d_f32, N * sizeof(float));
    __nv_fp8_e4m3 *d_e4m3; cudaMalloc(&d_e4m3, N * sizeof(__nv_fp8_e4m3));
    __nv_fp8_e5m2 *d_e5m2; cudaMalloc(&d_e5m2, N * sizeof(__nv_fp8_e5m2));
    __nv_bfloat16 *d_bf16; cudaMalloc(&d_bf16, N * sizeof(__nv_bfloat16));

    cudaMemset(d_f32, 0, N * sizeof(float));
    cudaMemset(d_bf16, 0, N * sizeof(__nv_bfloat16));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100;
    int blocks = 148, threads = 256;

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
        long total_ops = (long)N * iters;
        double gops = total_ops / (best/1000.0) / 1e9;
        printf("  %-30s %7.3f ms  %7.1f Gops/s\n", name, best, gops);
    };

    printf("# B300 FP8 conversion throughput\n");
    printf("# 1M elements × 100 iter convert\n\n");

    bench([&]{ cvt_f32_to_e4m3<<<blocks, threads>>>(d_e4m3, d_f32, N, iters); }, "F32 → E4M3");
    bench([&]{ cvt_f32_to_e5m2<<<blocks, threads>>>(d_e5m2, d_f32, N, iters); }, "F32 → E5M2");
    bench([&]{ cvt_bf16_to_e4m3<<<blocks, threads>>>(d_e4m3, d_bf16, N, iters); }, "BF16 → E4M3");
    bench([&]{ cvt_e4m3_to_f32<<<blocks, threads>>>(d_f32, d_e4m3, N, iters); }, "E4M3 → F32");

    return 0;
}
