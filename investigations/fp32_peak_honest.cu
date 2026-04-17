// FP32 FFMA peak — v2, fixes from v1:
// 1. b/c loaded at runtime to prevent compile-time folding
// 2. Correct unit conversion (GFLOPS → TFLOPS)
// 3. SASS-verified with nvcc -keep (manually)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 50000
#define ILP 16

extern "C" __global__ void k_peak(float *in_bc, float *out) {
    // Load b, c from memory — compiler can't fold
    float b = in_bc[0];
    float c = in_bc[1];

    float a[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; j++) a[j] = in_bc[2 + j] + threadIdx.x * 0.0001f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++)
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(a[j]) : "f"(b), "f"(c));
    }

    // Anti-DCE: unconditional write
    if (threadIdx.x == 0) {
        float s = 0;
        #pragma unroll
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

    // Runtime constants
    float h_bc[2 + ILP];
    h_bc[0] = 1.00001f;
    h_bc[1] = 0.00001f;
    for (int i = 0; i < ILP; i++) h_bc[2 + i] = 1.0f + i * 0.1f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, (2 + ILP) * sizeof(float));
    cudaMalloc(&d_out, blocks * sizeof(float));
    cudaMemcpy(d_in, h_bc, (2 + ILP) * sizeof(float), cudaMemcpyHostToDevice);

    cudaStream_t s; cudaStreamCreate(&s);

    // Warmup
    for (int i = 0; i < 5; i++) {
        k_peak<<<blocks, threads, 0, s>>>(d_in, d_out);
    }
    cudaDeviceSynchronize();

    // Measure
    float best_ms = 1e30f;
    for (int t = 0; t < 10; t++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        k_peak<<<blocks, threads, 0, s>>>(d_in, d_out);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
        if (ms < best_ms) best_ms = ms;
    }

    // FLOPs count
    long long ffmas = (long long)blocks * threads * ITERS * ILP;
    long long flops = ffmas * 2;  // 1 FMA = 2 FLOPS
    double tflops_measured = (double)flops / (best_ms / 1e3) / 1e12;

    // Theoretical (in GFLOPS then convert to TFLOPS)
    double theoretical_2032 = 148.0 * 128.0 * 2.0 * 2.032;  // GFLOPS
    double theoretical_2032_tflops = theoretical_2032 / 1000.0;
    double theoretical_1920_tflops = (148.0 * 128.0 * 2.0 * 1.920) / 1000.0;

    printf("# B300 FP32 FFMA peak (v2 — runtime b/c, proper unit handling)\n");
    printf("# Config: %d blocks × %d threads, ILP=%d, ITERS=%d, warps/SM=%d\n",
           blocks, threads, ILP, ITERS, threads/32 * blocks_per_sm);
    printf("#\n");
    printf("# Runtime: %.4f ms\n", best_ms);
    printf("# FFMAs executed: %lld\n", ffmas);
    printf("# FLOPS executed: %lld\n", flops);
    printf("# Measured: %.2f TFLOPS\n", tflops_measured);
    printf("#\n");
    printf("# Theoretical @ 2032 MHz (default): %.2f TFLOPS\n", theoretical_2032_tflops);
    printf("# Theoretical @ 1920 MHz (locked):  %.2f TFLOPS\n", theoretical_1920_tflops);
    printf("# Achievement vs 2032 MHz: %.1f%%\n", tflops_measured / theoretical_2032_tflops * 100);
    printf("#\n");
    if (tflops_measured > theoretical_2032_tflops * 1.05) {
        printf("# ERROR: measurement exceeds theoretical — probable DCE or formula bug\n");
    } else if (tflops_measured > theoretical_2032_tflops * 0.9) {
        printf("# Result: AT PEAK (>90%% theoretical)\n");
    } else if (tflops_measured > theoretical_2032_tflops * 0.5) {
        printf("# Result: plausible but under-saturated (50-90%% theoretical)\n");
    } else {
        printf("# Result: WAY under-saturated — investigate DCE, ILP, or occupancy\n");
    }

    cudaStreamDestroy(s);
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
