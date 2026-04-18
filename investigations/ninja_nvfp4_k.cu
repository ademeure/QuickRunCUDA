// Test K=96 hypothesis: NVFP4 with K specifically set to multiples of 96
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

void measure(int M, int N, int K, const char *label) {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    size_t a_bytes = (size_t)M*K/2, b_bytes = (size_t)K*N/2, c_bytes = (size_t)M*N*2;
    size_t a_scale_bytes = (size_t)M*K/16, b_scale_bytes = (size_t)K*N/16;
    void *d_a, *d_b, *d_c, *d_d, *d_a_scale, *d_b_scale, *d_ws;
    cudaMalloc(&d_a, a_bytes); cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes); cudaMalloc(&d_d, c_bytes);
    cudaMalloc(&d_a_scale, a_scale_bytes); cudaMalloc(&d_b_scale, b_scale_bytes);
    size_t ws = 256ull*1024*1024; cudaMalloc(&d_ws, ws);

    unsigned char *h = (unsigned char*)malloc(b_bytes);  // bigger of two
    srand(42);
    for (size_t i = 0; i < b_bytes; i++) h[i] = rand() & 0xff;
    cudaMemcpy(d_a, h, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_scale, h, a_scale_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_scale, h, b_scale_bytes, cudaMemcpyHostToDevice);
    free(h);

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

    // A is M×K (transposed for opT), B is K×N
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, K, M, K);  // K=lda, M=cols
    cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, M, N, M);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    if (st != 0 || nr == 0) {
        printf("  %s M=%d N=%d K=%d: NO HEUR\n", label, M, N, K);
        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
        cudaFree(d_a_scale); cudaFree(d_b_scale); cudaFree(d_ws);
        return;
    }

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
    printf("  %s M=%5d N=%5d K=%5d: %.4f ms = %.0f TFLOPS = %.1f%% of 10000 spec  (K%%96=%d, K%%64=%d)\n",
           label, M, N, K, best, tflops, tflops/10000*100, K%96, K%64);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d);
    cudaFree(d_a_scale); cudaFree(d_b_scale); cudaFree(d_ws);
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulPreferenceDestroy(pref); cudaStreamDestroy(s);
    cublasLtDestroy(lt);
}

int main() {
    printf("# NVFP4 K-sweep test: does K=multiple-of-96 unlock 1.5x throughput?\n\n");
    printf("# Vary K with fixed M=N=8192:\n");
    for (int K : {6144, 6240, 6336, 8160, 8192, 8256, 9216, 9600, 12096, 12288, 12480, 18432}) {
        measure(8192, 8192, K, "");
    }
    printf("\n# Vary K with fixed M=N=16384:\n");
    for (int K : {12288, 16384, 18432, 24576}) {
        measure(16384, 16384, K, "");
    }
    return 0;
}
