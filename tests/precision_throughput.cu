// FP throughput characterization: FP32, FP16, BF16
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <chrono>

#define ITERS 100000

extern "C" __global__ void k_fp32(float *out) {
    float a = 1.0f + threadIdx.x * 0.001f;
    float b = 1.0001f, c = 0.0001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(b), "f"(c));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_fp64(double *out) {
    double a = 1.0 + threadIdx.x * 0.001;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
        asm volatile("fma.rn.f64 %0,%0,%1,%2;" : "+d"(a) : "d"(1.0001), "d"(0.0001));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

extern "C" __global__ void k_fp16(__half *out) {
    __half2 a = __float2half2_rn(1.0f);
    __half2 b = __float2half2_rn(1.0001f);
    __half2 c = __float2half2_rn(0.0001f);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a = __hfma2(a, b, c);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = __low2half(a);
}

extern "C" __global__ void k_bf16(__nv_bfloat16 *out) {
    __nv_bfloat162 a = __float2bfloat162_rn(1.0f);
    __nv_bfloat162 b = __float2bfloat162_rn(1.0001f);
    __nv_bfloat162 c = __float2bfloat162_rn(0.0001f);
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        a = __hfma2(a, b, c);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = __low2bfloat16(a);
}

extern "C" __global__ void k_int32(int *out) {
    int a = threadIdx.x + 1;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++)
        asm volatile("mad.lo.s32 %0, %0, %1, %2;" : "+r"(a) : "r"(2), "r"(3));
    if (threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int blocks = prop.multiProcessorCount, threads = 128;

    float *d_out_f;
    double *d_out_d;
    __half *d_out_h;
    __nv_bfloat16 *d_out_bf;
    int *d_out_i;
    cudaMalloc(&d_out_f, blocks * sizeof(float));
    cudaMalloc(&d_out_d, blocks * sizeof(double));
    cudaMalloc(&d_out_h, blocks * sizeof(__half));
    cudaMalloc(&d_out_bf, blocks * sizeof(__nv_bfloat16));
    cudaMalloc(&d_out_i, blocks * sizeof(int));

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 2; i++) { fn(); cudaDeviceSynchronize(); }
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 precision throughput (148 blocks × 128 threads × %d iters)\n", ITERS);
    printf("# Each kernel: latency-bound FMA chain (1 dependent thread)\n\n");

    float t = bench([&]{ k_fp32<<<blocks, threads, 0, s>>>(d_out_f); });
    long long ops = (long long)blocks * threads * ITERS * 2;  // FMA = 2 ops
    printf("  FP32 FFMA: %.4f ms = %.2f TFLOPS\n", t, ops/(t/1e3)/1e12);

    t = bench([&]{ k_fp64<<<blocks, threads, 0, s>>>(d_out_d); });
    printf("  FP64 DFMA: %.4f ms = %.2f TFLOPS\n", t, ops/(t/1e3)/1e12);

    t = bench([&]{ k_fp16<<<blocks, threads, 0, s>>>(d_out_h); });
    long long ops_h2 = (long long)blocks * threads * ITERS * 4;  // half2 FMA = 4 ops (2 lanes × 2)
    printf("  FP16x2 HFMA: %.4f ms = %.2f TFLOPS\n", t, ops_h2/(t/1e3)/1e12);

    t = bench([&]{ k_bf16<<<blocks, threads, 0, s>>>(d_out_bf); });
    printf("  BF16x2 BFFMA: %.4f ms = %.2f TFLOPS\n", t, ops_h2/(t/1e3)/1e12);

    t = bench([&]{ k_int32<<<blocks, threads, 0, s>>>(d_out_i); });
    printf("  INT32 MAD:  %.4f ms = %.2f TIOPS\n", t, ops/(t/1e3)/1e12);

    cudaStreamDestroy(s);
    cudaFree(d_out_f); cudaFree(d_out_d); cudaFree(d_out_h); cudaFree(d_out_bf); cudaFree(d_out_i);
    return 0;
}
