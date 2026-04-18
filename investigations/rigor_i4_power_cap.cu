// I4 RIGOR: power cap behavior — measure FP8 cuBLAS at multiple caps
// CRITICAL: RESET to default (1100 W) at end via cleanup.
//
// Test data variants: ZERO, RANDOM, REALISTIC (e.g. normal distribution).
// Different inputs may cause different power draw and thus different
// throughput at a given cap.

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { auto e=(x); if (e!=cudaSuccess){printf("CUDA err %d\n",(int)e);exit(1);}} while(0)

int main(int argc, char **argv) {
    int N = 8192;
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    void *d_a, *d_b, *d_c, *d_d, *d_workspace;
    cudaMalloc(&d_a, (size_t)N*N);     // FP8 = 1B
    cudaMalloc(&d_b, (size_t)N*N);
    cudaMalloc(&d_c, (size_t)N*N*2);   // BF16 output = 2B
    cudaMalloc(&d_d, (size_t)N*N*2);
    size_t ws = 256ull * 1024 * 1024;
    cudaMalloc(&d_workspace, ws);

    // Init data based on argv[1]
    const char *mode = argc > 1 ? argv[1] : "zero";
    if (!strcmp(mode, "zero")) {
        cudaMemset(d_a, 0, (size_t)N*N);
        cudaMemset(d_b, 0, (size_t)N*N);
    } else if (!strcmp(mode, "byte")) {
        cudaMemset(d_a, 0x42, (size_t)N*N);
        cudaMemset(d_b, 0x42, (size_t)N*N);
    } else { // random or realistic — host-side init
        unsigned char *h = (unsigned char*)malloc((size_t)N*N);
        srand(42);
        for (size_t i = 0; i < (size_t)N*N; i++) h[i] = rand() & 0xff;
        cudaMemcpy(d_a, h, (size_t)N*N, cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, h, (size_t)N*N, cudaMemcpyHostToDevice);
        free(h);
    }

    // Setup matmul desc for FP8
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayout_t a_layout, b_layout, c_layout;
    cublasLtMatrixLayoutCreate(&a_layout, CUDA_R_8F_E4M3, N, N, N);
    cublasLtMatrixLayoutCreate(&b_layout, CUDA_R_8F_E4M3, N, N, N);
    cublasLtMatrixLayoutCreate(&c_layout, CUDA_R_16BF, N, N, N);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

    cublasLtMatmulHeuristicResult_t heur[1];
    int n_results;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a_layout, b_layout, c_layout, c_layout, pref, 1, heur, &n_results);
    if (n_results == 0) { printf("no heur\n"); return 1; }

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cublasLtMatmul(lt, desc, &alpha, d_a, a_layout, d_b, b_layout, &beta, d_c, c_layout, d_d, c_layout,
                       &heur[0].algo, d_workspace, ws, 0);
    }
    cudaDeviceSynchronize();

    // Measure
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        cublasLtMatmul(lt, desc, &alpha, d_a, a_layout, d_b, b_layout, &beta, d_c, c_layout, d_d, c_layout,
                       &heur[0].algo, d_workspace, ws, 0);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = 2L * N * N * N;
    double tflops = ops / (best/1000) / 1e12;
    printf("data=%s: best=%.4f ms = %.0f TFLOPS\n", mode, best, tflops);
    return 0;
}
