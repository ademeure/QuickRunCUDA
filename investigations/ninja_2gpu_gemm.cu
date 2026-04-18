// 2-GPU split GEMM: each GPU does half the rows of C
// Both GPUs have their own copy of B; A is split row-wise
// This is "model-parallel" tensor split (rows of A → rows of C)
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <chrono>

int main() {
    int N = 8192;
    int half = N / 2;  // each GPU processes 4096 rows
    long ops_total = 2L * N * N * N;
    long ops_half = ops_total / 2;

    int n_gpus; cudaGetDeviceCount(&n_gpus);
    if (n_gpus < 2) { printf("Need 2 GPUs\n"); return 1; }

    // Setup on both GPUs
    cublasLtHandle_t lt[2];
    void *dA[2], *dB[2], *dC[2], *dD[2], *dWS[2];
    cudaStream_t s[2];
    cudaEvent_t e0[2], e1[2];

    for (int g = 0; g < 2; g++) {
        cudaSetDevice(g);
        cublasLtCreate(&lt[g]);
        // Each GPU has half-A (4096 × 8192), full B (8192 × 8192), half-C (4096 × 8192)
        cudaMalloc(&dA[g], (size_t)half * N * 2);  // BF16
        cudaMalloc(&dB[g], (size_t)N * N * 2);
        cudaMalloc(&dC[g], (size_t)half * N * 2);
        cudaMalloc(&dD[g], (size_t)half * N * 2);
        cudaMemset(dA[g], 0x42, (size_t)half * N * 2);
        cudaMemset(dB[g], 0x42, (size_t)N * N * 2);
        cudaMalloc(&dWS[g], 256ull*1024*1024);
        cudaStreamCreate(&s[g]);
        cudaEventCreate(&e0[g]); cudaEventCreate(&e1[g]);
    }

    // Same matmul desc on both
    cublasLtMatmulDesc_t desc[2];
    cublasLtMatrixLayout_t lay_a[2], lay_b[2], lay_c[2];
    cublasLtMatmulHeuristicResult_t heur[2];
    for (int g = 0; g < 2; g++) {
        cudaSetDevice(g);
        cublasLtMatmulDescCreate(&desc[g], CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(desc[g], CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(desc[g], CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        // A is half × N (transposed) → effective shape K=half, M=N (this matches T-on-A)
        // Actually: C[half×N] = A[half×N]^T × B[N×N]? Let's just do half=M, N=N, K=N
        cublasLtMatrixLayoutCreate(&lay_a[g], CUDA_R_16BF, N, half, N);
        cublasLtMatrixLayoutCreate(&lay_b[g], CUDA_R_16BF, N, N, N);
        cublasLtMatrixLayoutCreate(&lay_c[g], CUDA_R_16BF, half, N, half);
        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        size_t ws = 256ull*1024*1024;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        int nr;
        cublasLtMatmulAlgoGetHeuristic(lt[g], desc[g], lay_a[g], lay_b[g], lay_c[g], lay_c[g],
            pref, 1, &heur[g], &nr);
        if (nr == 0) { printf("GPU %d: no heur\n", g); return 1; }
        cublasLtMatmulPreferenceDestroy(pref);
    }

    float alpha = 1, beta = 0;
    auto run = [&]() {
        for (int g = 0; g < 2; g++) {
            cudaSetDevice(g);
            cublasLtMatmul(lt[g], desc[g], &alpha, dA[g], lay_a[g], dB[g], lay_b[g],
                &beta, dC[g], lay_c[g], dD[g], lay_c[g], &heur[g].algo, dWS[g], 256ull*1024*1024, s[g]);
        }
    };

    // Warmup
    for (int i = 0; i < 5; i++) run();
    for (int g = 0; g < 2; g++) { cudaSetDevice(g); cudaStreamSynchronize(s[g]); }

    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaSetDevice(0); cudaEventRecord(e0[0], s[0]);
        cudaSetDevice(1); cudaEventRecord(e0[1], s[1]);
        run();
        cudaSetDevice(0); cudaEventRecord(e1[0], s[0]);
        cudaSetDevice(1); cudaEventRecord(e1[1], s[1]);
        cudaSetDevice(0); cudaEventSynchronize(e1[0]);
        cudaSetDevice(1); cudaEventSynchronize(e1[1]);
        float ms0, ms1;
        cudaSetDevice(0); cudaEventElapsedTime(&ms0, e0[0], e1[0]);
        cudaSetDevice(1); cudaEventElapsedTime(&ms1, e0[1], e1[1]);
        float ms = ms0 > ms1 ? ms0 : ms1;
        if (ms < best) best = ms;
    }

    double tflops_agg = ops_total / (best/1000) / 1e12;  // both GPUs together
    double tflops_per = ops_half / (best/1000) / 1e12;   // per GPU
    printf("# 2-GPU split BF16 GEMM N=%d (each GPU does %d×%d×%d)\n", N, half, N, N);
    printf("  Wall (max): %.4f ms\n", best);
    printf("  Aggregate: %.0f TFLOPS\n", tflops_agg);
    printf("  Per-GPU: %.0f TFLOPS\n", tflops_per);
    printf("  vs single-GPU 8K^3 cuBLAS (~0.49 ms = 2247 TFLOPS): speedup %.2fx\n",
           0.49 / best);
    return 0;
}
