// cuBLAS DGEMM peak measurement on B300 - never directly measured
// Theoretical:
//   - FP64 DFMA non-tensor: 1.20 TFLOPS (verified)
//   - FP64 DMMA (mma.sync m16n8k4): ~2 TFLOPS (catalog estimate, MED conf)
//   - B300 spec for FP64 tensor: TBD via measurement
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdio>

int main(int argc, char** argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 8192;
    int N = (argc > 2) ? atoi(argv[2]) : 8192;
    int K = (argc > 3) ? atoi(argv[3]) : 8192;

    cudaSetDevice(0);
    cublasHandle_t h; cublasCreate(&h);

    double *A, *B, *C;
    cudaMalloc(&A, (size_t)M*K*sizeof(double));
    cudaMalloc(&B, (size_t)K*N*sizeof(double));
    cudaMalloc(&C, (size_t)M*N*sizeof(double));
    cudaMemset(A, 0x42, (size_t)M*K*sizeof(double));
    cudaMemset(B, 0x42, (size_t)K*N*sizeof(double));

    double alpha = 1.0, beta = 0.0;
    cudaStream_t s; cudaStreamCreate(&s);
    cublasSetStream(h, s);

    // Warmup
    for (int i = 0; i < 3; i++) {
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
    }
    cudaStreamSynchronize(s);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0, s);
        cublasDgemm(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, A, M, B, K, &beta, C, M);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = 2L * M * N * K;
    double tflops = ops / (best/1000.0) / 1e12;
    printf("cuBLAS DGEMM M=%d N=%d K=%d: %.4f ms = %.2f TFLOPS\n", M, N, K, best, tflops);

    // Also try gemmEx with FP64 tensor compute
    cublasMath_t prev_math;
    cublasGetMathMode(h, &prev_math);
    cublasSetMathMode(h, CUBLAS_DEFAULT_MATH);

    cublasComputeType_t comp_types[] = {
        CUBLAS_COMPUTE_64F,
        CUBLAS_COMPUTE_64F_PEDANTIC,
    };
    const char* names[] = {
        "CUBLAS_COMPUTE_64F",
        "CUBLAS_COMPUTE_64F_PEDANTIC",
    };
    for (int t = 0; t < 2; t++) {
        for (int i = 0; i < 3; i++) {
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                A, CUDA_R_64F, M, B, CUDA_R_64F, K, &beta,
                C, CUDA_R_64F, M, comp_types[t], CUBLAS_GEMM_DEFAULT_TENSOR_OP);
        }
        cudaStreamSynchronize(s);
        float best2 = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaEventRecord(e0, s);
            cublasGemmEx(h, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha,
                A, CUDA_R_64F, M, B, CUDA_R_64F, K, &beta,
                C, CUDA_R_64F, M, comp_types[t], CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best2) best2 = ms;
        }
        double tflops2 = ops / (best2/1000.0) / 1e12;
        printf("cuBLAS GemmEx %s + TENSOR_OP: %.4f ms = %.2f TFLOPS\n", names[t], best2, tflops2);
    }

    return 0;
}
