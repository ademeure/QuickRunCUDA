// FP16/BF16 throughput - packed math via half2/bfloat162
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>

__global__ void hfma2_chain(__half2 *out, int iters, __half2 k1, __half2 k2) {
    __half2 a = __halves2half2(__float2half(threadIdx.x*0.001f), __float2half(threadIdx.x*0.002f));
    __half2 b = __halves2half2(__float2half(threadIdx.x*0.003f), __float2half(threadIdx.x*0.004f));
    __half2 c = __halves2half2(__float2half(threadIdx.x*0.005f), __float2half(threadIdx.x*0.006f));
    __half2 d = __halves2half2(__float2half(threadIdx.x*0.007f), __float2half(threadIdx.x*0.008f));
    for (int i = 0; i < iters; i++) {
        a = __hfma2(a, k1, k2);
        b = __hfma2(b, k1, k2);
        c = __hfma2(c, k1, k2);
        d = __hfma2(d, k1, k2);
    }
    __half2 sum = __hadd2(__hadd2(a,b), __hadd2(c,d));
    if (__half2float(sum.x) + __half2float(sum.y) < -1e30f)
        out[blockIdx.x] = sum;
}

__global__ void bfma2_chain(__nv_bfloat162 *out, int iters, __nv_bfloat162 k1, __nv_bfloat162 k2) {
    __nv_bfloat162 a = __halves2bfloat162(__float2bfloat16(threadIdx.x*0.001f), __float2bfloat16(threadIdx.x*0.002f));
    __nv_bfloat162 b = __halves2bfloat162(__float2bfloat16(threadIdx.x*0.003f), __float2bfloat16(threadIdx.x*0.004f));
    __nv_bfloat162 c = __halves2bfloat162(__float2bfloat16(threadIdx.x*0.005f), __float2bfloat16(threadIdx.x*0.006f));
    __nv_bfloat162 d = __halves2bfloat162(__float2bfloat16(threadIdx.x*0.007f), __float2bfloat16(threadIdx.x*0.008f));
    for (int i = 0; i < iters; i++) {
        a = __hfma2(a, k1, k2);
        b = __hfma2(b, k1, k2);
        c = __hfma2(c, k1, k2);
        d = __hfma2(d, k1, k2);
    }
    __nv_bfloat162 sum = __hadd2(__hadd2(a,b), __hadd2(c,d));
    if (__bfloat162float(sum.x) + __bfloat162float(sum.y) < -1e30f)
        out[blockIdx.x] = sum;
}

__global__ void ffma_chain(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    float b = threadIdx.x * 0.002f;
    float c = threadIdx.x * 0.003f;
    float d = threadIdx.x * 0.004f;
    for (int i = 0; i < iters; i++) {
        a = a*k1 + k2;
        b = b*k1 + k2;
        c = c*k1 + k2;
        d = d*k1 + k2;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x] = a+b+c+d;
}

__global__ void dfma_chain(double *out, int iters, double k1, double k2) {
    double a = threadIdx.x * 0.001;
    double b = threadIdx.x * 0.002;
    double c = threadIdx.x * 0.003;
    double d = threadIdx.x * 0.004;
    for (int i = 0; i < iters; i++) {
        a = a*k1 + k2;
        b = b*k1 + k2;
        c = c*k1 + k2;
        d = d*k1 + k2;
    }
    if (a+b+c+d < -1e30) out[blockIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    void *d_out; cudaMalloc(&d_out, 1024 * 8);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148, threads = 128;

    auto bench = [&](auto launch, int iters, int ops_per_iter, const char *name) {
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
        long total_ops = (long)blocks * threads * iters * 4 * ops_per_iter;
        double tflops = total_ops * 2.0 / (best/1000.0) / 1e12;  // FMA = 2 FLOPS
        printf("  %-15s %d iters: %7.3f ms, %6.1f TFLOPS\n", name, iters, best, tflops);
    };

    printf("# B300 FP precision throughput (4-ILP, 148 blocks × 128 threads)\n");
    printf("# Theoretical FFMA peak: 76.96 TFLOPS at 2032 MHz\n\n");

    // FP32
    {
        int iters = 100000;
        float k1 = 1.0001f, k2 = 0.0001f;
        bench([&]{ ffma_chain<<<blocks, threads>>>((float*)d_out, iters, k1, k2); },
              iters, 1, "FP32 FFMA");
    }

    // FP16 packed (half2 = 2 ops per FMA)
    {
        int iters = 100000;
        __half2 k1 = __floats2half2_rn(1.0001f, 1.0001f);
        __half2 k2 = __floats2half2_rn(0.0001f, 0.0001f);
        bench([&]{ hfma2_chain<<<blocks, threads>>>((__half2*)d_out, iters, k1, k2); },
              iters, 2, "FP16 HFMA2");
    }

    // BF16 packed
    {
        int iters = 100000;
        __nv_bfloat162 k1 = __floats2bfloat162_rn(1.0001f, 1.0001f);
        __nv_bfloat162 k2 = __floats2bfloat162_rn(0.0001f, 0.0001f);
        bench([&]{ bfma2_chain<<<blocks, threads>>>((__nv_bfloat162*)d_out, iters, k1, k2); },
              iters, 2, "BF16 BFMA2");
    }

    // FP64
    {
        int iters = 100000;
        double k1 = 1.0001, k2 = 0.0001;
        bench([&]{ dfma_chain<<<blocks, threads>>>((double*)d_out, iters, k1, k2); },
              iters, 1, "FP64 DFMA");
    }

    return 0;
}
