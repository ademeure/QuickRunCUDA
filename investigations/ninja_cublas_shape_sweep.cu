// B3: cuBLAS algo selection across shapes — where does cuBLAS pick differently?
// Sweep M=N=K from small to large, report algoId + TFLOPS for top heuristic.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

void run(cublasLtHandle_t lt, int M, int N, int K, void *d_a, void *d_b, void *d_c, void *d_d, void *d_ws, size_t ws, cudaStream_t s, cudaEvent_t e0, cudaEvent_t e1) {
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

    cublasLtMatmulHeuristicResult_t heur[8];
    int nr = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 8, heur, &nr);
    if (nr == 0) {
        printf("  M=%-5d N=%-5d K=%-5d  no algo\n", M, N, K);
        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulPreferenceDestroy(pref);
        return;
    }
    int algoId = -1, tileId = -1;
    cublasLtMatmulAlgoConfigGetAttribute(&heur[0].algo, CUBLASLT_ALGO_CONFIG_ID, &algoId, sizeof(algoId), nullptr);
    cublasLtMatmulAlgoConfigGetAttribute(&heur[0].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tileId, sizeof(tileId), nullptr);

    float alpha = 1, beta = 0;
    for (int i = 0; i < 5; i++) cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
    cudaStreamSynchronize(s);
    float best = 1e30f;
    int n_iter = (M < 1024) ? 100 : (M < 4096 ? 20 : 5);
    cudaEventRecord(e0, s);
    for (int i = 0; i < n_iter; i++) cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
    cudaEventRecord(e1, s); cudaStreamSynchronize(s);
    float ms; cudaEventElapsedTime(&ms, e0, e1); ms /= n_iter;
    long ops = 2L * M * N * K;
    double tflops = ops / (ms/1000.0) / 1e12;
    printf("  M=%-5d N=%-5d K=%-5d  algoId=%-4d  tileId=%-5d  ws=%-6zu KB  %.4f ms  %.1f TFLOPS  (%5.1f%% spec)\n",
           M, N, K, algoId, tileId, heur[0].workspaceSize/1024,
           ms, tflops, tflops/2.5e3*100);

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulPreferenceDestroy(pref);
}

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    cudaMalloc(&d_a, 16384L*16384*2);
    cudaMalloc(&d_b, 16384L*16384*2);
    cudaMalloc(&d_c, 16384L*16384*2);
    cudaMalloc(&d_d, 16384L*16384*2);
    size_t ws = 1024ull*1024*1024;
    cudaMalloc(&d_ws, ws);
    cudaMemset(d_a, 0x3c, 16384L*16384*2);
    cudaMemset(d_b, 0x3c, 16384L*16384*2);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# cuBLAS BF16 GEMM algo selection across shapes\n");
    printf("# Theoretical peak BF16 = 2500 TFLOPS\n\n");
    printf("# Square shapes:\n");
    for (int sz : {128, 256, 512, 1024, 2048, 4096, 8192, 16384}) run(lt, sz, sz, sz, d_a, d_b, d_c, d_d, d_ws, ws, s, e0, e1);

    printf("\n# Inference-decode shapes (M=1, K varies):\n");
    for (int k : {1024, 4096, 8192, 16384}) run(lt, 1, 4096, k, d_a, d_b, d_c, d_d, d_ws, ws, s, e0, e1);

    printf("\n# Tall-skinny shapes (M small, N=K large):\n");
    for (int m : {16, 64, 256, 1024}) run(lt, m, 8192, 8192, d_a, d_b, d_c, d_d, d_ws, ws, s, e0, e1);

    return 0;
}
