// Compare cuBLAS GEMM throughput across precisions on B300
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { if((c) != cudaSuccess) { printf("CUDA err: %d\n", c); exit(1); } } while(0)
#define CKCB(c) do { if((c) != CUBLAS_STATUS_SUCCESS) { printf("cuBLAS err: %d\n", c); exit(1); } } while(0)

struct GemmSpec {
    const char *name;
    cudaDataType_t A_type, B_type, C_type;
    cublasComputeType_t compute_type;
    int bytes_per_op;
};

void bench(cublasLtHandle_t lt_handle, cudaStream_t s, int M, int N, int K, GemmSpec spec) {
    // Allocate matrices
    void *A, *B, *C;
    cudaMalloc(&A, (size_t)M * K * spec.bytes_per_op);
    cudaMalloc(&B, (size_t)K * N * spec.bytes_per_op);
    cudaMalloc(&C, (size_t)M * N * 2);  // FP16/BF16 output = 2 bytes
    cudaMemset(A, 0x10, (size_t)M * K * spec.bytes_per_op);
    cudaMemset(B, 0x10, (size_t)K * N * spec.bytes_per_op);
    cudaMemset(C, 0, (size_t)M * N * 2);

    cublasLtMatmulDesc_t desc;
    CKCB(cublasLtMatmulDescCreate(&desc, spec.compute_type, CUDA_R_32F));
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N));

    cublasLtMatrixLayout_t Adesc, Bdesc, Cdesc;
    cublasLtMatrixLayoutCreate(&Adesc, spec.A_type, M, K, M);
    cublasLtMatrixLayoutCreate(&Bdesc, spec.B_type, K, N, K);
    cublasLtMatrixLayoutCreate(&Cdesc, spec.C_type, M, N, M);

    float alpha = 1.0f, beta = 0.0f;
    size_t workspace_sz = 32 * 1024 * 1024;
    void *workspace;
    cudaMalloc(&workspace, workspace_sz);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                          &workspace_sz, sizeof(workspace_sz));

    {
    cublasLtMatmulHeuristicResult_t heur;
    int returned = 0;
    cublasStatus_t h_st = cublasLtMatmulAlgoGetHeuristic(lt_handle, desc, Adesc, Bdesc, Cdesc, Cdesc,
                                                         pref, 1, &heur, &returned);
    if (h_st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        printf("  %-20s %4d³: NO ALGO (status %d returned %d)\n", spec.name, M, h_st, returned);
        goto cleanup;
    }

    // Warmup
    for (int i = 0; i < 3; i++) {
        CKCB(cublasLtMatmul(lt_handle, desc, &alpha, A, Adesc, B, Bdesc, &beta,
                            C, Cdesc, C, Cdesc, &heur.algo, workspace, workspace_sz, s));
    }
    cudaStreamSynchronize(s);

    // Measure
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(t0, s);
        CKCB(cublasLtMatmul(lt_handle, desc, &alpha, A, Adesc, B, Bdesc, &beta,
                            C, Cdesc, C, Cdesc, &heur.algo, workspace, workspace_sz, s));
        cudaEventRecord(t1, s);
        cudaEventSynchronize(t1);
        float ms; cudaEventElapsedTime(&ms, t0, t1);
        if (ms < best) best = ms;
    }
    cudaEventDestroy(t0); cudaEventDestroy(t1);

    double tflops = 2.0 * M * N * K / (best / 1e3) / 1e12;
    printf("  %-20s %4d³: %.3f ms, %.0f TFLOPS\n", spec.name, M, best, tflops);
    }

cleanup:
    cublasLtMatrixLayoutDestroy(Adesc);
    cublasLtMatrixLayoutDestroy(Bdesc);
    cublasLtMatrixLayoutDestroy(Cdesc);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatmulPreferenceDestroy(pref);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(workspace);
}

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt;
    cublasLtCreate(&lt);
    cudaStream_t s; cudaStreamCreate(&s);

    GemmSpec specs[] = {
        {"FP16x→FP16", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, 2},
        {"FP16→FP32", CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_32F, 2},
        {"BF16→FP32", CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, 2},
        {"FP8(E4M3)→FP16", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F, CUBLAS_COMPUTE_32F, 1},
        {"FP8(E4M3)→BF16", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, 1},
    };

    printf("# B300 cuBLAS GEMM across precisions\n");
    printf("# clock: "); fflush(stdout);
    system("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader | head -1");
    printf("\n");

    int sizes[] = {2048, 4096, 8192};
    for (int M : sizes) {
        printf("--- %d³ ---\n", M);
        for (auto &spec : specs) {
            bench(lt, s, M, M, M, spec);
        }
    }

    cublasLtDestroy(lt);
    return 0;
}
