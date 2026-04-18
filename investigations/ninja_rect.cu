// Verify: single-GPU rectangular NVFP4 — does N=K=16384 with M=8192 hit 9582?
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

void test(int M, int N, int K) {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    void *d_a, *d_b, *d_c, *d_d, *d_a_scale, *d_b_scale, *d_ws;
    cudaMalloc(&d_a, (size_t)M*K/2); cudaMalloc(&d_b, (size_t)K*N/2);
    cudaMalloc(&d_c, (size_t)M*N*2); cudaMalloc(&d_d, (size_t)M*N*2);
    cudaMalloc(&d_a_scale, (size_t)M*K/16); cudaMalloc(&d_b_scale, (size_t)K*N/16);
    size_t ws = 256ull*1024*1024; cudaMalloc(&d_ws, ws);
    cudaMemset(d_a, 0x42, (size_t)M*K/2); cudaMemset(d_b, 0x42, (size_t)K*N/2);
    cudaMemset(d_a_scale, 0x40, (size_t)M*K/16);
    cudaMemset(d_b_scale, 0x40, (size_t)K*N/16);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale));
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, K, M, K);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, M, N, M);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    if (nr == 0) { printf("M=%d N=%d K=%d: NO HEUR\n", M, N, K); return; }

    cudaStream_t s; cudaStreamCreate(&s);
    float alpha=1, beta=0;
    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaStreamSynchronize(s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0, s);
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = 2L * M * N * K;
    double tflops = ops / (best/1000) / 1e12;
    printf("  M=%5d N=%5d K=%5d: %.4f ms = %.0f TFLOPS = %.1f%% of 10000 spec\n",
           M, N, K, best, tflops, tflops/10000*100);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    cudaFree(d_a_scale); cudaFree(d_b_scale); cudaFree(d_ws);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulPreferenceDestroy(pref); cudaStreamDestroy(s);
    cublasLtDestroy(lt);
}

int main() {
    printf("# NVFP4 shape hunt — push beyond 10000 TFLOPS toward 15000 (B300 1.5x)\n");
    // Wide N (output tile huge)
    test( 8192, 32768, 16384);  // 100.6% prior best
    test( 8192, 65536, 16384);
    test( 8192, 49152, 16384);
    test( 4096, 65536, 16384);
    test(16384, 32768, 16384);
    // Different K
    test( 8192, 32768, 24576);
    test( 8192, 32768, 32768);
    // Very wide
    test( 4096, 98304, 16384);
    return 0;
}
