// PEAK FP32 throughput with ILP unrolling
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 50000
#define ILP 8

extern "C" __global__ void k_peak_ilp(float *out) {
    float a0 = 1.0f + threadIdx.x * 0.0001f;
    float a1 = 2.0f + threadIdx.x * 0.0002f;
    float a2 = 3.0f + threadIdx.x * 0.0003f;
    float a3 = 4.0f + threadIdx.x * 0.0004f;
    float a4 = 5.0f + threadIdx.x * 0.0005f;
    float a5 = 6.0f + threadIdx.x * 0.0006f;
    float a6 = 7.0f + threadIdx.x * 0.0007f;
    float a7 = 8.0f + threadIdx.x * 0.0008f;
    float b = 1.0001f, c = 0.0001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a0) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a1) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a2) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a3) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a4) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a5) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a6) : "f"(b), "f"(c));
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a7) : "f"(b), "f"(c));
    }
    if (threadIdx.x == 0) out[blockIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    // Saturate: 148 SMs × multiple blocks each
    int blocks = prop.multiProcessorCount * 4, threads = 256;

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

    float t = bench([&]{ k_peak_ilp<<<blocks, threads, 0, s>>>(d_out); });
    long long ops = (long long)blocks * threads * ITERS * ILP * 2;
    float tflops = ops/(t/1e3)/1e12;
    printf("# B300 FP32 PEAK with ILP=%d, threads=128/block, %d blocks, %d iter\n",
           ILP, blocks, ITERS);
    printf("  Time: %.4f ms\n", t);
    printf("  Ops: %lld (= %d block × %d thr × %d iter × %d ILP × 2 op/FMA)\n",
           ops, blocks, threads, ITERS, ILP);
    printf("  Throughput: %.2f TFLOPS\n", tflops);

    cudaStreamDestroy(s);
    cudaFree(d_out);
    return 0;
}
