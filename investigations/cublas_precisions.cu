// cuBLAS GEMM at various precisions
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);

    int N = 8192;

    printf("# B300 cuBLAS precision sweep at M=N=K=%d\n\n", N);

    // FP32 (already measured but for comparison)
    {
        float *A, *B, *C;
        size_t bytes = (size_t)N * N * sizeof(float);
        cudaMalloc(&A, bytes); cudaMalloc(&B, bytes); cudaMalloc(&C, bytes);
        cudaMemset(A, 0, bytes); cudaMemset(B, 0, bytes);

        float alpha = 1.0f, beta = 0.0f;
        for (int i = 0; i < 3; i++)
            cublasSgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
        cudaDeviceSynchronize();

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
        double tflops = 2.0*N*N*N/(best/1000)/1e12;
        printf("  Sgemm (FP32):    %.3f ms = %.1f TFLOPS\n", best, tflops);

        cudaFree(A); cudaFree(B); cudaFree(C);
    }

    // FP16
    {
        __half *A, *B, *C;
        size_t bytes = (size_t)N * N * sizeof(__half);
        cudaMalloc(&A, bytes); cudaMalloc(&B, bytes); cudaMalloc(&C, bytes);
        cudaMemset(A, 0, bytes); cudaMemset(B, 0, bytes);

        __half alpha = __float2half(1.0f), beta = __float2half(0.0f);
        for (int i = 0; i < 3; i++)
            cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            cublasHgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = 2.0*N*N*N/(best/1000)/1e12;
        printf("  Hgemm (FP16):    %.3f ms = %.1f TFLOPS\n", best, tflops);

        cudaFree(A); cudaFree(B); cudaFree(C);
    }

    // FP64
    {
        double *A, *B, *C;
        size_t bytes = (size_t)N * N * sizeof(double);
        cudaMalloc(&A, bytes); cudaMalloc(&B, bytes); cudaMalloc(&C, bytes);
        cudaMemset(A, 0, bytes); cudaMemset(B, 0, bytes);

        double alpha = 1.0, beta = 0.0;
        for (int i = 0; i < 3; i++)
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
        cudaDeviceSynchronize();

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, N, N, N, &alpha, A, N, B, N, &beta, C, N);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = 2.0*N*N*N/(best/1000)/1e12;
        printf("  Dgemm (FP64):    %.3f ms = %.1f TFLOPS\n", best, tflops);

        cudaFree(A); cudaFree(B); cudaFree(C);
    }

    return 0;
}
