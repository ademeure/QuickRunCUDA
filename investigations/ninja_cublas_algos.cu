// A5: enumerate ALL cuBLAS LtMatmul algos for FP8 N=8192, identify which use tcgen05
// Hypothesis: tcgen05 algos hit 4400 TFLOPS, mma.sync legacy ones cap at ~570

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <vector>
#include <algorithm>

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
    cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, N, N, N);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, N, N, N);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, N, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    // Get up to 32 heuristic algos
    cublasLtMatmulHeuristicResult_t heur[32];
    int n_results = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 32, heur, &n_results);
    printf("# cuBLAS heuristic returned %d algorithms for FP8 e4m3 N=%d\n", n_results, N);

    float alpha = 1, beta = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    long ops = 2L * N * N * N;

    printf("# rank | algo_id | tile_id | warps | time(ms) | TFLOPS\n");
    for (int ai = 0; ai < n_results; ai++) {
        // Get algo id and tile id from this algo
        int algo_id = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&heur[ai].algo, CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), nullptr);
        int tile_id = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&heur[ai].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(tile_id), nullptr);

        // Warmup
        for (int i = 0; i < 3; i++) {
            cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[ai].algo, d_ws, ws, 0);
        }
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("  %2d | %3d | %3d | -- | ERROR: %s\n", ai, algo_id, tile_id, cudaGetErrorString(err));
            continue;
        }

        // Time
        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaEventRecord(e0);
            cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[ai].algo, d_ws, ws, 0);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = ops / (best/1000) / 1e12;
        printf("  %2d | %3d | %3d | %.4f ms | %.0f TFLOPS\n",
               ai, algo_id, tile_id, best, tflops);
    }

    return 0;
}
