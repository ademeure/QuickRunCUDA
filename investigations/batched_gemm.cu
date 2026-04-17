// Strided batched GEMM throughput
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);

    printf("# B300 cuBLAS strided batched Sgemm\n");
    printf("# Per-batch FLOPs = 2*N*N*N\n\n");
    printf("# %-8s %-8s %-12s %-15s\n", "N", "batch", "time_ms", "TFLOPS");

    for (int N : {64, 128, 256, 512, 1024, 2048}) {
        for (int batch : {1, 16, 256, 4096, 65536}) {
            size_t a_per = (size_t)N * N;
            size_t total_size = a_per * batch * sizeof(float);
            if (total_size * 3 > 100ull * 1024 * 1024 * 1024) continue;

            float *A, *B, *C;
            cudaMalloc(&A, total_size); cudaMalloc(&B, total_size); cudaMalloc(&C, total_size);
            cudaMemset(A, 0, total_size);
            cudaMemset(B, 0, total_size);

            float alpha = 1.0f, beta = 0.0f;
            for (int i = 0; i < 3; i++) {
                cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha,
                    A, N, a_per,
                    B, N, a_per, &beta,
                    C, N, a_per, batch);
            }
            cudaDeviceSynchronize();

            cudaEvent_t e0, e1;
            cudaEventCreate(&e0); cudaEventCreate(&e1);
            float best = 1e30f;
            for (int i = 0; i < 5; i++) {
                cudaEventRecord(e0);
                cublasSgemmStridedBatched(h, CUBLAS_OP_N, CUBLAS_OP_N,
                    N, N, N, &alpha,
                    A, N, a_per,
                    B, N, a_per, &beta,
                    C, N, a_per, batch);
                cudaEventRecord(e1);
                cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) best = ms;
            }
            double total_flops = 2.0 * N * N * N * batch;
            double tflops = total_flops / (best/1000) / 1e12;
            printf("  %-8d %-8d %-12.3f %-15.1f\n", N, batch, best, tflops);

            cudaFree(A); cudaFree(B); cudaFree(C);
        }
    }

    return 0;
}
