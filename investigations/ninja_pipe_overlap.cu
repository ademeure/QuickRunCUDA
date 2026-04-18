// Test if FFMA pipe + XU/MUFU pipe can run concurrently at full rate
//
// Theoretical:
//   FFMA pipe: 1 inst/cy/SMSP * 32 lanes = 32 FFMAs/cy/SMSP = 64 FLOPs/cy/SMSP
//     Per SM: 4 SMSPs * 64 = 256 FLOPs/cy = 0.5 TFLOPs at 2.032 GHz
//     Per chip: 148 SMs * 0.5 TF = 74.0 TFLOPS theoretical
//   MUFU pipe: 0.5 inst/cy/SMSP * 32 lanes = 16 ex2/cy/SMSP
//     Per chip: 0.5 * 32 * 4 SMSPs * 148 SMs * 2.032e9 = 19.24 Tex2/s
//   If parallel: BOTH at full rate. Test shows interaction.
#include <cuda_runtime.h>
#include <cstdio>

// FFMA-only
template <int ILP>
__launch_bounds__(256, 8) __global__ void ffma_only(float *out, int N) {
    float a = (float)threadIdx.x * 0.001f;
    float b = (float)blockIdx.x * 0.0003f + 1.0f;
    float regs[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) regs[i] = (float)(threadIdx.x * (i+1));
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) regs[j] = regs[j] * a + b;
    }
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ILP; i++) sum += regs[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

// MUFU-only (ex2)
template <int ILP>
__launch_bounds__(256, 8) __global__ void mufu_only(float *out, int N) {
    float regs[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) regs[i] = (float)threadIdx.x * 0.001f + i * 0.0001f;
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(regs[j]));
        }
    }
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ILP; i++) sum += regs[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

// MIXED: FFMA + MUFU interleaved (alternating)
template <int ILP>
__launch_bounds__(256, 8) __global__ void mixed(float *out, int N) {
    float a = (float)threadIdx.x * 0.001f;
    float b = (float)blockIdx.x * 0.0003f + 1.0f;
    float fma_regs[ILP], mufu_regs[ILP];
    #pragma unroll
    for (int i = 0; i < ILP; i++) {
        fma_regs[i]  = (float)(threadIdx.x * (i+1));
        mufu_regs[i] = (float)(threadIdx.x * (i+1)) * 0.0001f;
    }
    #pragma unroll 1
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++) {
            fma_regs[j] = fma_regs[j] * a + b;
            asm volatile("ex2.approx.ftz.f32 %0, %0;" : "+f"(mufu_regs[j]));
        }
    }
    float sum = 0;
    #pragma unroll
    for (int i = 0; i < ILP; i++) sum += fma_regs[i] + mufu_regs[i];
    if (sum == 0xdeadbeef) out[blockIdx.x] = sum;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * 1024);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int blocks = 148 * 8, threads = 256;
    int N = 50000;

    auto bench = [&](const char* name, void(*kfn)(float*, int), int ffma_per_iter, int mufu_per_iter) {
        for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N);
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) return;
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            kfn<<<blocks, threads>>>(d_out, N);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long total_ffma = (long)blocks * threads * N * ffma_per_iter;
        long total_mufu = (long)blocks * threads * N * mufu_per_iter;
        double ffma_tflops = total_ffma * 2.0 / (best/1000.0) / 1e12;
        double mufu_gops = total_mufu / (best/1000.0) / 1e9;
        printf("  %-25s %.4f ms  FFMA=%.1f TFLOPS  MUFU=%.0f Gex2/s\n",
            name, best, ffma_tflops, mufu_gops);
    };

    printf("# Pipe-overlap test (B300 sm_103a, OCC=8, blocks=%d)\n", blocks);
    bench("FFMA only ILP=8",   ffma_only<8>, 8, 0);
    bench("MUFU only ILP=8",   mufu_only<8>, 0, 8);
    bench("MIXED ILP=8 (each)", mixed<8>,    8, 8);
    return 0;
}
