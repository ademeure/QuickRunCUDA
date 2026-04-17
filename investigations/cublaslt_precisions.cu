// cuBLASLt for various precisions
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cudaStream_t s; cudaStreamCreate(&s);

    int M = 8192, N = 8192, K = 8192;

    auto run_gemm = [&](cudaDataType_t a_type, cudaDataType_t b_type, cudaDataType_t c_type,
                        cublasComputeType_t comp_type, const char *name) {
        size_t a_bytes = (size_t)M * K * (a_type == CUDA_R_8F_E4M3 || a_type == CUDA_R_8F_E5M2 ? 1 :
                                          (a_type == CUDA_R_16F || a_type == CUDA_R_16BF ? 2 : 4));
        size_t b_bytes = (size_t)K * N * (b_type == CUDA_R_8F_E4M3 || b_type == CUDA_R_8F_E5M2 ? 1 :
                                          (b_type == CUDA_R_16F || b_type == CUDA_R_16BF ? 2 : 4));
        size_t c_bytes = (size_t)M * N * (c_type == CUDA_R_16F || c_type == CUDA_R_16BF ? 2 : 4);
        size_t ws_sz = 64 * 1024 * 1024;

        void *A, *B, *C, *ws;
        cudaMalloc(&A, a_bytes); cudaMalloc(&B, b_bytes); cudaMalloc(&C, c_bytes); cudaMalloc(&ws, ws_sz);

        cublasLtMatmulDesc_t desc;
        cublasLtMatmulDescCreate(&desc, comp_type, c_type);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        cublasLtMatrixLayout_t Ad, Bd, Cd;
        cublasLtMatrixLayoutCreate(&Ad, a_type, M, K, M);
        cublasLtMatrixLayoutCreate(&Bd, b_type, K, N, K);
        cublasLtMatrixLayoutCreate(&Cd, c_type, M, N, M);

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz));
        cublasLtMatmulHeuristicResult_t heur;
        int returned;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref, 1, &heur, &returned);
        if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
            printf("  %-30s NO ALGO (status %d)\n", name, st);
            return;
        }

        float alpha = 1.0f, beta = 0.0f;

        // Warmup
        for (int i = 0; i < 3; i++)
            cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
        cudaStreamSynchronize(s);

        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s);
            cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = 2.0 * M * N * K / (best/1000.0) / 1e12;
        printf("  %-30s %.2f ms = %.1f TFLOPS\n", name, best, tflops);

        cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ws);
    };

    printf("# B300 cuBLASLt at M=N=K=%d\n\n", M);

    run_gemm(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F, "FP32 (TF32 disabled)");
    run_gemm(CUDA_R_32F, CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, "FP32 → TF32");
    run_gemm(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUBLAS_COMPUTE_16F, "FP16");
    run_gemm(CUDA_R_16F, CUDA_R_16F, CUDA_R_32F, CUBLAS_COMPUTE_32F, "FP16 in / FP32 acc");
    run_gemm(CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F, "BF16 in/out, FP32 acc");
    run_gemm(CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, "FP8 E4M3 in / BF16 out");
    run_gemm(CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_16BF, CUBLAS_COMPUTE_32F, "FP8 E5M2 in / BF16 out");

    cublasLtDestroy(lt);
    return 0;
}
