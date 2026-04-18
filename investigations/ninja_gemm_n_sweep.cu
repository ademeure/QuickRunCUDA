// FP8 cuBLAS scaling vs N — does it stay near 4400 TFLOPS at all sizes?
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    size_t max_bytes = (size_t)32768 * 32768;
    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    cudaMalloc(&d_a, max_bytes); cudaMemset(d_a, 0x42, max_bytes);
    cudaMalloc(&d_b, max_bytes); cudaMemset(d_b, 0x42, max_bytes);
    cudaMalloc(&d_c, max_bytes * 2);
    cudaMalloc(&d_d, max_bytes * 2);
    size_t ws = 256ull*1024*1024;
    cudaMalloc(&d_ws, ws);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float alpha=1, beta=0;

    printf("# FP8 cuBLAS LtMatmul scaling, square M=N=K\n");
    printf("# N      ms        TFLOPS    %% of 5000 NVIDIA spec\n");

    int Ns[] = {1024, 2048, 4096, 6144, 8192, 10240, 12288, 16384, 24576, 32768};
    for (int N : Ns) {
        cublasLtMatmulDesc_t desc;
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatrixLayout_t a, b, c;
        cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, N, N, N);
        cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, N, N, N);
        cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, N, N);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        cublasLtMatmulHeuristicResult_t heur[1]; int nr;
        cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
        if (nr == 0) { printf("  N=%5d: NO HEUR\n", N); continue; }

        for (int i = 0; i < 3; i++)
            cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, 0);
        cudaDeviceSynchronize();

        float best = 1e30f;
        int reps = (N <= 4096) ? 100 : (N <= 12288 ? 30 : 10);
        for (int i = 0; i < reps; i++) {
            cudaEventRecord(e0);
            cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, 0);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = 2L * N * N * N;
        double tflops = ops / (best/1000) / 1e12;
        printf("  N=%5d: %.4f ms = %.0f TFLOPS = %.1f%%\n", N, best, tflops, tflops/5000*100);
        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(a);
        cublasLtMatrixLayoutDestroy(b);
        cublasLtMatrixLayoutDestroy(c);
        cublasLtMatmulPreferenceDestroy(pref);
    }
    return 0;
}
