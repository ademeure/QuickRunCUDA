// A5: Enumerate cuBLAS algorithms for BF16 GEMM and identify fastest
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

int main(int argc, char**argv) {
    cudaSetDevice(0);
    int M = 8192, N = 8192, K = 8192;
    if (argc > 1) M = N = K = atoi(argv[1]);

    cublasLtHandle_t lt; cublasLtCreate(&lt);
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

    constexpr int MAX_ALGOS = 32;
    cublasLtMatmulHeuristicResult_t heur[MAX_ALGOS];
    int nr = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, MAX_ALGOS, heur, &nr);
    printf("# BF16 GEMM M=N=K=%d, %d algorithms returned by heuristic\n", M, nr);
    printf("# rank  algoId  tile  stages  splitK  swizzle  ws_KB    waves   TFLOPS\n");

    float alpha = 1, beta = 0;
    long ops = 2L * M * N * K;

    for (int i = 0; i < nr; i++) {
        int algoId = -1, tileId = -1, stages = -1, splitK = -1, swizzle = -1;
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages, sizeof(stages), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &splitK, sizeof(splitK), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swizzle, sizeof(swizzle), nullptr);

        for (int j = 0; j < 3; j++) cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[i].algo, d_ws, ws, s);
        cudaStreamSynchronize(s);

        float best = 1e30f;
        for (int j = 0; j < 3; j++) {
            cudaEventRecord(e0, s);
            cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[i].algo, d_ws, ws, s);
            cudaEventRecord(e1, s); cudaStreamSynchronize(s);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = ops / (best/1000.0) / 1e12;
        printf("  %2d    %4d    %4d   %3d     %3d     %3d     %6zu   %.2f    %.1f\n",
               i, algoId, tileId, stages, splitK, swizzle,
               heur[i].workspaceSize / 1024, heur[i].wavesCount, tflops);
    }
    return 0;
}
