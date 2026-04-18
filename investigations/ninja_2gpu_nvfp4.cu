// 2-GPU split NVFP4 GEMM — each GPU does half the rows
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main() {
    int N = 16384, half = N/2;
    int K = N;  // K=N=16384

    int n_gpus; cudaGetDeviceCount(&n_gpus);
    if (n_gpus < 2) { printf("Need 2 GPUs\n"); return 1; }

    cublasLtHandle_t lt[2];
    void *dA[2], *dB[2], *dC[2], *dD[2], *dWS[2], *dAS[2], *dBS[2];
    cudaStream_t s[2]; cudaEvent_t e0[2], e1[2];

    for (int g = 0; g < 2; g++) {
        cudaSetDevice(g);
        cublasLtCreate(&lt[g]);
        // Half-A is half × K, full-B is K × N, half-C is half × N
        cudaMalloc(&dA[g], (size_t)half * K / 2);
        cudaMalloc(&dB[g], (size_t)K * N / 2);
        cudaMalloc(&dC[g], (size_t)half * N * 2);
        cudaMalloc(&dD[g], (size_t)half * N * 2);
        cudaMalloc(&dAS[g], (size_t)half * K / 16);
        cudaMalloc(&dBS[g], (size_t)K * N / 16);
        cudaMalloc(&dWS[g], 256ull*1024*1024);
        cudaMemset(dA[g], 0x42, (size_t)half*K/2);
        cudaMemset(dB[g], 0x42, (size_t)K*N/2);
        cudaMemset(dAS[g], 0x40, (size_t)half*K/16);
        cudaMemset(dBS[g], 0x40, (size_t)K*N/16);
        cudaStreamCreate(&s[g]);
        cudaEventCreate(&e0[g]); cudaEventCreate(&e1[g]);
    }

    cublasLtMatmulDesc_t desc[2];
    cublasLtMatrixLayout_t la[2], lb[2], lc[2];
    cublasLtMatmulHeuristicResult_t heur[2];
    for (int g = 0; g < 2; g++) {
        cudaSetDevice(g);
        cublasLtMatmulDescCreate(&desc[g], CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(desc[g], CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(desc[g], CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulDescSetAttribute(desc[g], CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
        cublasLtMatmulDescSetAttribute(desc[g], CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
        cublasLtMatmulDescSetAttribute(desc[g], CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &dAS[g], sizeof(dAS[g]));
        cublasLtMatmulDescSetAttribute(desc[g], CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &dBS[g], sizeof(dBS[g]));
        cublasLtMatrixLayoutCreate(&la[g], CUDA_R_4F_E2M1, K, half, K);
        cublasLtMatrixLayoutCreate(&lb[g], CUDA_R_4F_E2M1, K, N, K);
        cublasLtMatrixLayoutCreate(&lc[g], CUDA_R_16BF, half, N, half);
        cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
        size_t ws = 256ull*1024*1024;
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
        int nr;
        cublasLtMatmulAlgoGetHeuristic(lt[g], desc[g], la[g], lb[g], lc[g], lc[g], pref, 1, &heur[g], &nr);
        if (nr == 0) { printf("GPU %d: no heur\n", g); return 1; }
        cublasLtMatmulPreferenceDestroy(pref);
    }

    float alpha=1, beta=0;
    auto run = [&]() {
        for (int g = 0; g < 2; g++) {
            cudaSetDevice(g);
            cublasLtMatmul(lt[g], desc[g], &alpha, dA[g], la[g], dB[g], lb[g], &beta,
                dC[g], lc[g], dD[g], lc[g], &heur[g].algo, dWS[g], 256ull*1024*1024, s[g]);
        }
    };

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
    long ops_total = 2L * N * N * K;
    long ops_half = ops_total / 2;
    double tflops_agg = ops_total / (best/1000) / 1e12;
    double tflops_per = ops_half / (best/1000) / 1e12;
    printf("# 2-GPU NVFP4 GEMM, M=N=K=%d (each GPU does %d×%d×%d)\n", N, half, N, K);
    printf("  Wall (max): %.4f ms\n", best);
    printf("  Aggregate: %.0f TFLOPS\n", tflops_agg);
    printf("  Per-GPU:   %.0f TFLOPS\n", tflops_per);
    printf("  Theoretical 2-GPU max NVFP4: 20000 TFLOPS (= 2 × 10000)\n");
    printf("  Efficiency vs 2-GPU theoretical: %.1f%%\n", tflops_agg/20000*100);
    return 0;
}
