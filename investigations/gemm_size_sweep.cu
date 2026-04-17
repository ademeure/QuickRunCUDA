// Find peak throughput across GEMM dimensions for cuBLAS FP8 and BF16
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

void bench_one(cublasLtHandle_t lt, cudaStream_t s, int M, int N, int K,
               cudaDataType_t in_t, cudaDataType_t out_t, const char *label) {
    void *A, *B, *C;
    int in_bpe = (in_t == CUDA_R_8F_E4M3) ? 1 : 2;
    int out_bpe = (out_t == CUDA_R_16F || out_t == CUDA_R_16BF) ? 2 : 4;
    cudaMalloc(&A, (size_t)M*K*in_bpe);
    cudaMalloc(&B, (size_t)K*N*in_bpe);
    cudaMalloc(&C, (size_t)M*N*out_bpe);
    cudaMemset(A, 0x10, (size_t)M*K*in_bpe);
    cudaMemset(B, 0x10, (size_t)K*N*in_bpe);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayout_t Ad, Bd, Cd;
    cublasLtMatrixLayoutCreate(&Ad, in_t, M, K, M);
    cublasLtMatrixLayoutCreate(&Bd, in_t, K, N, K);
    cublasLtMatrixLayoutCreate(&Cd, out_t, M, N, M);

    float alpha = 1.0f, beta = 0.0f;
    size_t ws_sz = 64 * 1024 * 1024;
    void *ws; cudaMalloc(&ws, ws_sz);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz));

    cublasLtMatmulHeuristicResult_t heur;
    int returned;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref, 1, &heur, &returned);
    if (st != CUBLAS_STATUS_SUCCESS || returned == 0) {
        printf("  %s M=%-5d N=%-5d K=%-5d: NO ALGO\n", label, M, N, K);
        goto cleanup;
    }
    {
        for (int i = 0; i < 3; i++)
            cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
        cudaDeviceSynchronize();

        cudaEvent_t t0, t1;
        cudaEventCreate(&t0); cudaEventCreate(&t1);
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(t0, s);
            cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
            cudaEventRecord(t1, s);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            if (ms < best) best = ms;
        }
        double tflops = 2.0 * M * N * K / (best/1e3) / 1e12;
        printf("  %s M=%-5d N=%-5d K=%-5d: %.3f ms, %.0f TFLOPS\n",
               label, M, N, K, best, tflops);
        cudaEventDestroy(t0); cudaEventDestroy(t1);
    }

cleanup:
    cublasLtMatrixLayoutDestroy(Ad);
    cublasLtMatrixLayoutDestroy(Bd);
    cublasLtMatrixLayoutDestroy(Cd);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatmulPreferenceDestroy(pref);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ws);
}

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cudaStream_t s; cudaStreamCreate(&s);

    int sizes[] = {512, 1024, 2048, 4096, 8192, 12288, 16384};

    printf("# B300 cuBLAS GEMM size sweep — find peak\n\n");

    printf("## BF16 → FP32:\n");
    for (int sz : sizes) {
        bench_one(lt, s, sz, sz, sz, CUDA_R_16BF, CUDA_R_16BF, "BF16");
    }

    printf("\n## FP8 → BF16:\n");
    for (int sz : sizes) {
        bench_one(lt, s, sz, sz, sz, CUDA_R_8F_E4M3, CUDA_R_16BF, "FP8 ");
    }

    // Also try non-square (LLM-style: tall thin)
    printf("\n## BF16 LLM-decode style (M=1, large N, large K):\n");
    int Ks[] = {2048, 4096, 8192, 14336};  // typical hidden sizes
    for (int K : Ks) {
        bench_one(lt, s, 1, K * 4, K, CUDA_R_16BF, CUDA_R_16BF, "BF16-decode");
    }

    printf("\n## BF16 LLM-prefill style (M=4096, K=2048-14336):\n");
    for (int K : Ks) {
        bench_one(lt, s, 4096, K * 4, K, CUDA_R_16BF, CUDA_R_16BF, "BF16-prefill");
    }

    cublasLtDestroy(lt);
    return 0;
}
