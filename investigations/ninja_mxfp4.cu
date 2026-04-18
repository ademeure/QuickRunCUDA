// Try MX-scaled FP4 via cuBLAS LtMatmul descriptor
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

int main() {
    int N = 8192;
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    // FP4 = 4 bits/elem = N*N/2 bytes per matrix
    // MX-FP4 also needs scale matrices (1 scale per 32 elements typically)
    void *d_a, *d_b, *d_c, *d_d, *d_a_scale, *d_b_scale, *d_ws;
    cudaMalloc(&d_a, (size_t)N*N/2);
    cudaMalloc(&d_b, (size_t)N*N/2);
    cudaMalloc(&d_c, (size_t)N*N*2);
    cudaMalloc(&d_d, (size_t)N*N*2);
    // Scales: 1 ue8m0 per 32 elements → N*N/32 bytes per scale matrix
    size_t scale_bytes = (size_t)N*N/32;
    cudaMalloc(&d_a_scale, scale_bytes);
    cudaMalloc(&d_b_scale, scale_bytes);
    cudaMemset(d_a, 0x33, (size_t)N*N/2);
    cudaMemset(d_b, 0x33, (size_t)N*N/2);
    cudaMemset(d_a_scale, 0x42, scale_bytes);
    cudaMemset(d_b_scale, 0x42, scale_bytes);
    size_t ws = 256ull*1024*1024;
    cudaMalloc(&d_ws, ws);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    // Set MX scale mode
    // Try VEC16_UE4M3 (NVFP4 mode, more common for B300)
    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale));

    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, N, N, N);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, N, N, N);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, N, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    printf("MX-FP4 e2m1 N=%d: heur status=%d, n=%d\n", N, (int)st, nr);
    if (st != CUBLAS_STATUS_SUCCESS || nr == 0) return 1;

    cudaStream_t s; cudaStreamCreate(&s);
    float alpha=1, beta=0;
    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaError_t err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(err)); return 1; }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(e0, s);
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = 2L * N * N * N;
    double tflops = ops / (best/1000) / 1e12;
    printf("MX-FP4 e2m1: best %.4f ms = %.0f TFLOPS\n", best, tflops);
    return 0;
}
