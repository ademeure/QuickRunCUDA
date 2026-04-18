// Try NVFP4 + bias to unlock K=96 cuBLAS kernels
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

void test(int M, int N, int K, bool with_bias) {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    size_t a_bytes = (size_t)M*K/2, b_bytes = (size_t)K*N/2, c_bytes = (size_t)M*N*2;
    void *d_a, *d_b, *d_c, *d_d, *d_a_scale, *d_b_scale, *d_bias, *d_ws;
    cudaMalloc(&d_a, a_bytes); cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes); cudaMalloc(&d_d, c_bytes);
    cudaMalloc(&d_a_scale, (size_t)M*K/16);
    cudaMalloc(&d_b_scale, (size_t)K*N/16);
    cudaMalloc(&d_bias, M * 2);  // bias is per output row
    size_t ws = 256ull*1024*1024; cudaMalloc(&d_ws, ws);

    unsigned char *h = (unsigned char*)malloc(b_bytes);
    srand(42);
    for (size_t i = 0; i < b_bytes; i++) h[i] = rand() & 0xff;
    cudaMemcpy(d_a, h, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_scale, h, (size_t)M*K/16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_scale, h, (size_t)K*N/16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h, M * 2, cudaMemcpyHostToDevice);
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

    if (with_bias) {
        cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;  // BIAS+GELU (matches sm103 kernel names)
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias));
        cudaDataType_t bias_dtype = CUDA_R_16BF;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dtype, sizeof(bias_dtype));
    }

    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, K, M, K);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, M, N, M);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    if (st != 0 || nr == 0) {
        printf("  M=%d N=%d K=%d bias=%d: NO HEUR (st=%d)\n", M, N, K, with_bias, (int)st);
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
    printf("  M=%5d N=%5d K=%5d bias=%d: %.4f ms = %.0f TFLOPS = %.1f%% spec  (K%%96=%d)\n",
           M, N, K, with_bias, best, tflops, tflops/10000*100, K%96);
}

int main() {
    printf("# NVFP4 with bias: does it unlock K=96 kernels (1.5x throughput)?\n\n");
    printf("## Square N=K=M:\n");
    for (int N : {8192, 12288, 16384, 24576}) {
        test(N, N, N, false);
        test(N, N, N, true);
    }
    printf("\n## K=multiple-of-96 specifically:\n");
    test(8192, 8192, 12288, true);
    test(8192, 8192, 18432, true);
    test(16384, 16384, 12288, true);
    test(16384, 16384, 18432, true);
    return 0;
}
