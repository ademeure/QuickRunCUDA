// True FP32 peak with full ILP × occupancy
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 50000
#define ILP 16

extern "C" __global__ void k_fp32(float *out) {
    float a[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; j++) a[j] = 1.0f + threadIdx.x * 0.0001f * (j + 1);
    float b = 1.0001f, c = 0.0001f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++)
            asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a[j]) : "f"(b), "f"(c));
    }
    if (threadIdx.x == 0) {
        float s = 0;
        for (int j = 0; j < ILP; j++) s += a[j];
        out[blockIdx.x] = s;
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    int sm = prop.multiProcessorCount;
    int threads = 1024;
    int blocks_per_sm = 2;
    int blocks = sm * blocks_per_sm;

    float *d_out;
    cudaMalloc(&d_out, blocks * sizeof(float));

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

    float t = bench([&]{ k_fp32<<<blocks, threads, 0, s>>>(d_out); });
    long long ops = (long long)blocks * threads * ITERS * ILP * 2;
    float tflops = ops/(t/1e3)/1e12;

    printf("# B300 FP32 PEAK with ILP=%d, threads=%d/block, %d blocks, %d iter\n",
           ILP, threads, blocks, ITERS);
    printf("# Warps per SM: %d × %d = %d\n", threads/32, blocks_per_sm, threads/32 * blocks_per_sm);
    printf("# Time: %.4f ms\n", t);
    printf("# Throughput: %.2f TFLOPS\n", tflops);
    printf("# Theoretical peak: 4 partitions × 32 lanes × 1 FFMA/cy × 2 op/FMA × 148 SM × 2.032 GHz = 76.96 TFLOPS\n");
    printf("# Achievement: %.1f%% of peak\n", tflops/76.96*100);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
