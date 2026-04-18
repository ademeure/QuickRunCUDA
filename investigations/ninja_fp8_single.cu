// Single FP8 cuBLAS call for ncu profiling
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

int main() {
    int N = 8192;
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    cudaMalloc(&d_a, (size_t)N*N); cudaMemset(d_a, 0x42, (size_t)N*N);
    cudaMalloc(&d_b, (size_t)N*N); cudaMemset(d_b, 0x42, (size_t)N*N);
    cudaMalloc(&d_c, (size_t)N*N*2);
    cudaMalloc(&d_d, (size_t)N*N*2);
    size_t ws = 256ull * 1024 * 1024;
    cudaMalloc(&d_ws, ws);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, N, N, N);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, N, N, N);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, N, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);

    float alpha = 1, beta = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, 0);
    cudaDeviceSynchronize();

    float best = 1e30f;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(e0);
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = 2L * N * N * N;
    double tflops = ops / (best/1000) / 1e12;
    printf("# Single FP8 cuBLAS N=%d: %.4f ms = %.0f TFLOPS\n", N, best, tflops);
    return 0;
}
