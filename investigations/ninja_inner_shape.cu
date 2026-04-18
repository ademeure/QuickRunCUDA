// Enumerate INNER_SHAPE_ID for FP4 algos to find K=96
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>

int main() {
    int M=8192, N=65536, K=16384;  // The 10297 TFLOPS shape
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    void *d_a, *d_b, *d_c, *d_d, *d_a_scale, *d_b_scale, *d_ws;
    cudaMalloc(&d_a, (size_t)M*K/2); cudaMalloc(&d_b, (size_t)K*N/2);
    cudaMalloc(&d_c, (size_t)M*N*2); cudaMalloc(&d_d, (size_t)M*N*2);
    cudaMalloc(&d_a_scale, (size_t)M*K/16); cudaMalloc(&d_b_scale, (size_t)K*N/16);
    size_t ws = 256ull*1024*1024; cudaMalloc(&d_ws, ws);
    cudaMemset(d_a, 0x42, (size_t)M*K/2); cudaMemset(d_b, 0x42, (size_t)K*N/2);
    cudaMemset(d_a_scale, 0x40, (size_t)M*K/16); cudaMemset(d_b_scale, 0x40, (size_t)K*N/16);

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
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, K, M, K);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, M, N, M);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[8]; int nr;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 8, heur, &nr);

    // Print all heur algos including INNER_SHAPE_ID
    for (int i = 0; i < nr; i++) {
        int aid=0, tid=0, csid=0, isid=0, sid=0, sk=0, swi=0;
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_ID, &aid, sizeof(aid), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tid, sizeof(tid), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID, &csid, sizeof(csid), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_INNER_SHAPE_ID, &isid, sizeof(isid), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_STAGES_ID, &sid, sizeof(sid), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &sk, sizeof(sk), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur[i].algo, CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &swi, sizeof(swi), nullptr);
        printf("heur[%d] algo=%d tile=%d cluster=%d inner=%d stages=%d split_k=%d swiz=%d ws=%zu\n",
               i, aid, tid, csid, isid, sid, sk, swi, heur[i].workspaceSize);
    }

    // Now run heur[0] for baseline
    float alpha=1, beta=0;
    cudaStream_t s; cudaStreamCreate(&s);
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
    printf("\nHeur[0] baseline: %.0f TFLOPS\n", ops/(best/1000)/1e12);

    // Sweep STAGES_ID
    printf("\n# Sweep STAGES_ID 30-50 (heur uses 37-38):\n");
    for (int sid = 30; sid <= 50; sid++) {
        cublasLtMatmulAlgo_t algo = heur[0].algo;
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_STAGES_ID,
            &sid, sizeof(sid));
        cublasLtMatmulHeuristicResult_t check; memset(&check, 0, sizeof(check));
        cublasStatus_t cst = cublasLtMatmulAlgoCheck(lt, desc, a, b, c, c, &algo, &check);
        if (cst != 0) continue;
        cublasStatus_t st = cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta,
            d_c, c, d_d, c, &algo, d_ws, ws, s);
        cudaError_t err = cudaStreamSynchronize(s);
        if (st != 0 || err != cudaSuccess) { cudaGetLastError(); continue; }
        float local_best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s);
            cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta,
                d_c, c, d_d, c, &algo, d_ws, ws, s);
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < local_best) local_best = ms;
        }
        double tflops = ops / (local_best/1000) / 1e12;
        printf("  stages=%2d: %.4f ms = %.0f TFLOPS%s\n", sid, local_best, tflops,
               tflops > 13000 ? " <-- BREAKTHROUGH" : "");
    }
    return 0;
}
