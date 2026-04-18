// D4: Per-precision power/perf table — measure all major math precisions
// 1) FP32 FFMA peak (compute + power)
// 2) FP64 DFMA peak (compute + power)
// 3) BF16 mma.sync peak (compute, power from prior)
// 4) FP8 cuBLAS peak (compute + power via separate run)
// (FP4 covered separately in cuBLAS sweep)
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

__device__ __forceinline__ void mma_b16(unsigned (&d)[4],
                                        unsigned (&a)[4], unsigned (&b)[2],
                                        unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// FP8 mma.sync m16n8k32 e4m3
__device__ __forceinline__ void mma_fp8(unsigned (&d)[4],
                                        unsigned (&a)[4], unsigned (&b)[2],
                                        unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

// FP32 FFMA
__launch_bounds__(512, 1) __global__ void k_ffma(int *out, int N) {
    float a0 = threadIdx.x * 1.001f, b0 = 0.999f, c0 = 0.001f;
    float a1 = threadIdx.x * 1.002f, b1 = 0.998f, c1 = 0.002f;
    float a2 = threadIdx.x * 1.003f, b2 = 0.997f, c2 = 0.003f;
    float a3 = threadIdx.x * 1.004f, b3 = 0.996f, c3 = 0.004f;
    float a4 = threadIdx.x * 1.005f, b4 = 0.995f, c4 = 0.005f;
    float a5 = threadIdx.x * 1.006f, b5 = 0.994f, c5 = 0.006f;
    float a6 = threadIdx.x * 1.007f, b6 = 0.993f, c6 = 0.007f;
    float a7 = threadIdx.x * 1.008f, b7 = 0.992f, c7 = 0.008f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            a0 = a0 * b0 + c0;  a1 = a1 * b1 + c1;
            a2 = a2 * b2 + c2;  a3 = a3 * b3 + c3;
            a4 = a4 * b4 + c4;  a5 = a5 * b5 + c5;
            a6 = a6 * b6 + c6;  a7 = a7 * b7 + c7;
        }
    }
    if (a0+a1+a2+a3+a4+a5+a6+a7 == 0.0f && N < 0) out[threadIdx.x] = 1;
}

// FP64 DFMA
__launch_bounds__(512, 1) __global__ void k_dfma(int *out, int N) {
    double a0 = threadIdx.x * 1.001, b0 = 0.999, c0 = 0.001;
    double a1 = threadIdx.x * 1.002, b1 = 0.998, c1 = 0.002;
    double a2 = threadIdx.x * 1.003, b2 = 0.997, c2 = 0.003;
    double a3 = threadIdx.x * 1.004, b3 = 0.996, c3 = 0.004;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            a0 = a0 * b0 + c0;  a1 = a1 * b1 + c1;
            a2 = a2 * b2 + c2;  a3 = a3 * b3 + c3;
        }
    }
    if (a0+a1+a2+a3 == 0.0 && N < 0) out[threadIdx.x] = 1;
}

template <int THREADS>
__launch_bounds__(THREADS, 1) __global__ void k_mma_bf16(int *out, int N) {
    unsigned a0[4] = {0x3f800001, 0x3f800002, 0x3f800003, 0x3f800004};
    unsigned a1[4] = {0x3f800005, 0x3f800006, 0x3f800007, 0x3f800008};
    unsigned a2[4] = {0x3f800009, 0x3f80000a, 0x3f80000b, 0x3f80000c};
    unsigned a3[4] = {0x3f80000d, 0x3f80000e, 0x3f80000f, 0x3f800010};
    unsigned b0[2] = {0x3f800001, 0x3f800002};
    unsigned b1[2] = {0x3f800003, 0x3f800004};
    unsigned b2[2] = {0x3f800005, 0x3f800006};
    unsigned b3[2] = {0x3f800007, 0x3f800008};
    unsigned c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_b16(c0, a0, b0, c0);
            mma_b16(c1, a1, b1, c1);
            mma_b16(c2, a2, b2, c2);
            mma_b16(c3, a3, b3, c3);
        }
    }
    if (c0[0] == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0];
}

template <int THREADS>
__launch_bounds__(THREADS, 1) __global__ void k_mma_fp8(int *out, int N) {
    // Init from threadIdx so values are not compile-time constants
    unsigned a0[4] = {threadIdx.x | 0x40404040u, threadIdx.x | 0x41414141u, threadIdx.x | 0x42424242u, threadIdx.x | 0x43434343u};
    unsigned a1[4] = {threadIdx.x | 0x44444444u, threadIdx.x | 0x45454545u, threadIdx.x | 0x46464646u, threadIdx.x | 0x47474747u};
    unsigned a2[4] = {threadIdx.x | 0x48484848u, threadIdx.x | 0x49494949u, threadIdx.x | 0x4a4a4a4au, threadIdx.x | 0x4b4b4b4bu};
    unsigned a3[4] = {threadIdx.x | 0x4c4c4c4cu, threadIdx.x | 0x4d4d4d4du, threadIdx.x | 0x4e4e4e4eu, threadIdx.x | 0x4f4f4f4fu};
    unsigned b0[2] = {threadIdx.x | 0x40404040u, threadIdx.x | 0x41414141u};
    unsigned b1[2] = {threadIdx.x | 0x42424242u, threadIdx.x | 0x43434343u};
    unsigned b2[2] = {threadIdx.x | 0x44444444u, threadIdx.x | 0x45454545u};
    unsigned b3[2] = {threadIdx.x | 0x46464646u, threadIdx.x | 0x47474747u};
    unsigned c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_fp8(c0, a0, b0, c0);
            mma_fp8(c1, a1, b1, c1);
            mma_fp8(c2, a2, b2, c2);
            mma_fp8(c3, a3, b3, c3);
        }
    }
    // Force write to defeat DCE
    out[blockIdx.x * THREADS + threadIdx.x] = c0[0] + c1[0] + c2[0] + c3[0];
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](const char* name, auto launch, double ops_per_call, double spec) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s\n", cudaGetErrorString(cudaGetLastError())); return; }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = ops_per_call / (best/1000.0) / 1e12;
        printf("  %-25s  %.3f ms  %.1f TFLOPS  (%.1f%% of %.0f spec)\n",
               name, best, tflops, tflops/spec*100, spec);
    };

    int N = 1000;
    int blocks = 148, threads = 512;
    long thr = (long)blocks * threads;

    printf("# B300 Per-precision peak compute (mma.sync paths, 16 warps/SM=512 thr/block)\n\n");

    // FP32 FFMA: 8 chains × 64 inner × N_iter × thr × 2 ops/FFMA
    bench("FP32 FFMA",  [&]() { k_ffma<<<blocks, 512>>>(d_out, N); },
          (double)thr * N * N_INNER * 8 * 2, 76.0);
    // FP64 DFMA: 4 chains
    bench("FP64 DFMA",  [&]() { k_dfma<<<blocks, 512>>>(d_out, N); },
          (double)thr * N * N_INNER * 4 * 2, 1.2);
    // BF16 mma.sync m16n8k16 = 4096 ops/inst, 4 chains
    bench("BF16 mma.sync m16n8k16",  [&]() { k_mma_bf16<512><<<blocks, 512>>>(d_out, N); },
          (double)blocks * (512/32) * N * N_INNER * 4 * 4096, 2500.0);
    // FP8 mma.sync m16n8k32 = 8192 ops/inst, 4 chains
    bench("FP8 mma.sync m16n8k32",  [&]() { k_mma_fp8<512><<<blocks, 512>>>(d_out, N); },
          (double)blocks * (512/32) * N * N_INNER * 4 * 8192, 5000.0);

    return 0;
}
