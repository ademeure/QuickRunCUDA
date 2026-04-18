// C3: cuBLAS warmup characterization
//
// First call to cublasLtMatmul does heuristic selection, JIT cache lookup, etc.
// Subsequent calls should be faster. Quantify the curve.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <chrono>

#define CHK(x) do { auto e = (x); if (e != CUBLAS_STATUS_SUCCESS) { fprintf(stderr, "cuBLAS err %s:%d %d\n", __FILE__, __LINE__, e); exit(1); } } while(0)

int main(int argc, char**argv) {
    cudaSetDevice(0);
    int M = 4096, N = 4096, K = 4096;
    if (argc > 1) M = N = K = atoi(argv[1]);

    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    cudaMalloc(&d_a, (size_t)M*K*2);
    cudaMalloc(&d_b, (size_t)K*N*2);
    cudaMalloc(&d_c, (size_t)M*N*2);
    cudaMalloc(&d_d, (size_t)M*N*2);
    size_t ws = 1024ull*1024*1024;
    cudaMalloc(&d_ws, ws);
    cudaMemset(d_a, 0x3c, (size_t)M*K*2);
    cudaMemset(d_b, 0x3c, (size_t)K*N*2);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Wall-clock timer for first-handle creation
    auto t0 = std::chrono::high_resolution_clock::now();
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    auto t1 = std::chrono::high_resolution_clock::now();
    double create_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("# cuBLAS handle create: %.1f us\n", create_us);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, M, N, M);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    // Heuristic timing
    cublasLtMatmulHeuristicResult_t heur[8];
    int nr = 0;
    t0 = std::chrono::high_resolution_clock::now();
    cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 8, heur, &nr);
    t1 = std::chrono::high_resolution_clock::now();
    double heur_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("# Heuristic selection: %.1f us, %d algorithms returned\n", heur_us, nr);

    // First matmul timing (includes any JIT/setup)
    float alpha = 1, beta = 0;

    printf("# First N matmul calls timing (BF16 GEMM M=N=K=%d):\n", M);
    printf("# call#   wall_us    event_ms  TFLOPS\n");
    for (int i = 0; i < 20; i++) {
        cudaStreamSynchronize(s);
        auto wt0 = std::chrono::high_resolution_clock::now();
        cudaEventRecord(e0, s);
        cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
        cudaEventRecord(e1, s);
        cudaStreamSynchronize(s);
        auto wt1 = std::chrono::high_resolution_clock::now();
        double wall_us = std::chrono::duration<double, std::micro>(wt1 - wt0).count();
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        long ops = 2L * M * N * K;
        double tflops = ops / (ms/1000.0) / 1e12;
        printf("  %3d    %8.1f   %.4f   %.1f\n", i+1, wall_us, ms, tflops);
    }

    return 0;
}
