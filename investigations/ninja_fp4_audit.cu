// FP4 audit: characterize FP4 cuBLAS on B300
// Check: does cuBLAS support FP4 e2m1 GEMM? Throughput? Power?
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>

void try_fp4(int N, cudaDataType_t a_type, cudaDataType_t b_type,
             cudaDataType_t c_type, cublasComputeType_t comp_kind,
             const char *label, int a_bytes_per_2_elem, int c_bytes_per_2_elem) {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    // FP4 = half byte per element
    size_t a_bytes = (size_t)N*N * a_bytes_per_2_elem / 2;
    size_t c_bytes = (size_t)N*N * c_bytes_per_2_elem / 2;
    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    cudaMalloc(&d_a, a_bytes); cudaMalloc(&d_b, a_bytes);
    cudaMalloc(&d_c, c_bytes); cudaMalloc(&d_d, c_bytes);
    size_t ws = 256ull*1024*1024;
    cudaMalloc(&d_ws, ws);

    // Random init
    unsigned char *h = (unsigned char*)malloc(a_bytes);
    srand(42);
    for (size_t i = 0; i < a_bytes; i++) h[i] = rand() & 0xff;
    cudaMemcpy(d_a, h, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h, a_bytes, cudaMemcpyHostToDevice);
    free(h);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, comp_kind, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, a_type, N, N, N);
    cublasLtMatrixLayoutCreate(&b, b_type, N, N, N);
    cublasLtMatrixLayoutCreate(&c, c_type, N, N, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    if (st != CUBLAS_STATUS_SUCCESS || nr == 0) {
        printf("%s N=%d: NO HEUR (st=%d)\n", label, N, (int)st);
        return;
    }

    cudaStream_t s; cudaStreamCreate(&s);
    float alpha=1, beta=0;
    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaError_t err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) { printf("%s N=%d: warmup err %s\n", label, N, cudaGetErrorString(err)); return; }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0, s);
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = 2L * N * N * N;
    double tflops = ops / (best/1000) / 1e12;
    printf("%-30s N=%5d: per-call %.4f ms = %.0f TFLOPS\n", label, N, best, tflops);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d); cudaFree(d_ws);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulPreferenceDestroy(pref);
    cudaStreamDestroy(s);
    cublasLtDestroy(lt);
}

int main() {
    nvmlInit();
    printf("# FP4/FP6/MX/INT8 cuBLAS audit on B300\n\n");

    int N = 8192;
    // FP6 e3m2
    try_fp4(N, CUDA_R_6F_E3M2, CUDA_R_6F_E3M2, CUDA_R_16BF, CUBLAS_COMPUTE_32F, "FP6 e3m2", 2, 4);
    try_fp4(N, CUDA_R_6F_E2M3, CUDA_R_6F_E2M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, "FP6 e2m3", 2, 4);
    // FP4 e2m1
    try_fp4(N, CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16BF, CUBLAS_COMPUTE_32F, "FP4 e2m1", 1, 4);
    // INT8
    try_fp4(N, CUDA_R_8I, CUDA_R_8I, CUDA_R_32I, CUBLAS_COMPUTE_32I, "INT8 → INT32", 2, 8);

    nvmlShutdown();
    return 0;
}
