// cuBLAS Sgemm performance vs problem size
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);

    // Test square GEMM at various sizes
    printf("# B300 cuBLAS Sgemm (FP32 GEMM) performance vs N\n");
    printf("# Theoretical FFMA peak: 76.96 TFLOPS\n\n");
    printf("# %-8s %-12s %-12s %-12s\n", "N", "time_ms", "TFLOPS", "% peak");

    int sizes[] = {256, 512, 1024, 2048, 4096, 8192, 16384};
    for (int N : sizes) {
        size_t bytes = (size_t)N * N * sizeof(float);
        if (bytes * 3 > 100ull*1024*1024*1024) continue;

        float *A, *B, *C;
        cudaMalloc(&A, bytes);
        cudaMalloc(&B, bytes);
        cudaMalloc(&C, bytes);
        cudaMemset(A, 0, bytes);
        cudaMemset(B, 0, bytes);

        float alpha = 1.0f, beta = 0.0f;

        // Warmup
        for (int i = 0; i < 3; i++)
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
        cudaDeviceSynchronize();

        // Bench (best of 5)
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double flops = 2.0 * N * N * N;
        double tflops = flops / (best/1000.0) / 1e12;
        printf("  %-8d %-12.3f %-12.1f %-12.1f%%\n",
               N, best, tflops, tflops/76.96*100);
        cudaFree(A); cudaFree(B); cudaFree(C);
    }

    cublasDestroy(h);
    return 0;
}
