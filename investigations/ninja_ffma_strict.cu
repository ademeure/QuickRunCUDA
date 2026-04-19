// FFMA strict anti-DCE re-verify
#include <cuda_runtime.h>
#include <cstdio>
constexpr int N_INNER = 64;

// V1: S1-style (constants, impossible-cond write)
__launch_bounds__(512, 1) __global__ void k_v1(int *out, int N) {
    float a0 = 1.001f, b0 = 0.999f, c0 = 0.001f;
    float a1 = 1.002f, b1 = 0.998f, c1 = 0.002f;
    float a2 = 1.003f, b2 = 0.997f, c2 = 0.003f;
    float a3 = 1.004f, b3 = 0.996f, c3 = 0.004f;
    float a4 = 1.005f, b4 = 0.995f, c4 = 0.005f;
    float a5 = 1.006f, b5 = 0.994f, c5 = 0.006f;
    float a6 = 1.007f, b6 = 0.993f, c6 = 0.007f;
    float a7 = 1.008f, b7 = 0.992f, c7 = 0.008f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            a0 = a0*b0+c0; a1 = a1*b1+c1; a2 = a2*b2+c2; a3 = a3*b3+c3;
            a4 = a4*b4+c4; a5 = a5*b5+c5; a6 = a6*b6+c6; a7 = a7*b7+c7;
        }
    }
    if ((a0+a1+a2+a3+a4+a5+a6+a7) == 0.0f && N < 0) out[threadIdx.x] = 1;
}

// V2: Strict — thread-derived inputs, always-write output
__launch_bounds__(512, 1) __global__ void k_v2(float *out, int N) {
    float t = threadIdx.x;
    float a0 = t*1.001f, b0 = 0.999f, c0 = 0.001f;
    float a1 = t*1.002f, b1 = 0.998f, c1 = 0.002f;
    float a2 = t*1.003f, b2 = 0.997f, c2 = 0.003f;
    float a3 = t*1.004f, b3 = 0.996f, c3 = 0.004f;
    float a4 = t*1.005f, b4 = 0.995f, c4 = 0.005f;
    float a5 = t*1.006f, b5 = 0.994f, c5 = 0.006f;
    float a6 = t*1.007f, b6 = 0.993f, c6 = 0.007f;
    float a7 = t*1.008f, b7 = 0.992f, c7 = 0.008f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            a0 = a0*b0+c0; a1 = a1*b1+c1; a2 = a2*b2+c2; a3 = a3*b3+c3;
            a4 = a4*b4+c4; a5 = a5*b5+c5; a6 = a6*b6+c6; a7 = a7*b7+c7;
        }
    }
    out[blockIdx.x * 512 + threadIdx.x] = a0+a1+a2+a3+a4+a5+a6+a7;
}

int main() {
    cudaSetDevice(0);
    int *outi; cudaMalloc(&outi, 148*512*sizeof(int));
    float *outf; cudaMalloc(&outf, 148*512*sizeof(float));
    int N = 1000;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long thr = 148L * 512;
        long total_ffma = thr * (long)N * N_INNER * 8;  // 8 chains × 1 FFMA
        long total_flops = total_ffma * 2;  // FFMA = 2 FLOPS
        double tflops = total_flops / (best/1000.0) / 1e12;
        printf("  %-25s  %.3f ms  %.1f TFLOPS  (%.1f%% of 76 spec)\n",
               name, best, tflops, tflops/76*100);
    };
    bench("V1: S1-style", [&](){k_v1<<<148, 512>>>(outi, N);});
    bench("V2: Strict",   [&](){k_v2<<<148, 512>>>(outf, N);});
    return 0;
}
