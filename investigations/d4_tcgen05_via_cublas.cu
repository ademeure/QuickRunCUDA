// D4 RIGOR: tcgen05 actual peak via cuBLAS LtMatmul (which uses tcgen05 internally on B300)
// Catalog claims: FP4=9856, FP8=4486, BF16=2325 TFLOPS — verify or refute.
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

#define CHECK(x) do { auto e = (x); if (e != cudaSuccess) { printf("CUDA err %d at %s:%d\n", (int)e, __FILE__, __LINE__); exit(1); } } while(0)
#define LT_CHECK(x) do { auto e = (x); if (e != CUBLAS_STATUS_SUCCESS) { printf("cuBLASLt err %d at %s:%d\n", (int)e, __FILE__, __LINE__); exit(1); } } while(0)

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    int N = 8192;  // M = N = K
    long ops_per_gemm = 2L * N * N * N;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    void *d_a, *d_b, *d_c, *d_d, *d_workspace;
    CHECK(cudaMalloc(&d_a, (long)N*N*2));
    CHECK(cudaMalloc(&d_b, (long)N*N*2));
    CHECK(cudaMalloc(&d_c, (long)N*N*2));
    CHECK(cudaMalloc(&d_d, (long)N*N*2));
    size_t workspace_size = 256ull * 1024 * 1024;
    CHECK(cudaMalloc(&d_workspace, workspace_size));
    cudaMemset(d_a, 0x42, (long)N*N*2);
    cudaMemset(d_b, 0x42, (long)N*N*2);
    cudaMemset(d_c, 0x42, (long)N*N*2);

    auto bench = [&](const char* label, cudaDataType_t a_type, cudaDataType_t b_type,
                     cudaDataType_t c_type, cublasComputeType_t compute) {
        cublasLtMatmulDesc_t desc;
        LT_CHECK(cublasLtMatmulDescCreate(&desc, compute, CUDA_R_32F));
        cublasOperation_t opT = CUBLAS_OP_T;
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        cublasLtMatrixLayout_t a_layout, b_layout, c_layout;
        cublasLtMatrixLayoutCreate(&a_layout, a_type, N, N, N);  // K x M, transposed
        cublasLtMatrixLayoutCreate(&b_layout, b_type, N, N, N);
        cublasLtMatrixLayoutCreate(&c_layout, c_type, N, N, N);

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &workspace_size, sizeof(workspace_size));

        cublasLtMatmulHeuristicResult_t heur[1];
        int n_results = 0;
        cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, a_layout, b_layout,
                                                           c_layout, c_layout, pref, 1, heur, &n_results);
        if (st != CUBLAS_STATUS_SUCCESS || n_results == 0) {
            printf("  %s: NO heuristic available (st=%d, n=%d)\n", label, (int)st, n_results);
            cublasLtMatmulDescDestroy(desc);
            cublasLtMatrixLayoutDestroy(a_layout);
            cublasLtMatrixLayoutDestroy(b_layout);
            cublasLtMatrixLayoutDestroy(c_layout);
            cublasLtMatmulPreferenceDestroy(pref);
            return;
        }

        float alpha = 1.0f, beta = 0.0f;
        // Warmup
        for (int i = 0; i < 3; i++) {
            cublasLtMatmul(lt, desc, &alpha, d_a, a_layout, d_b, b_layout,
                          &beta, d_c, c_layout, d_d, c_layout,
                          &heur[0].algo, d_workspace, workspace_size, 0);
        }
        cudaDeviceSynchronize();

        float best = 1e30f;
        for (int i = 0; i < 10; i++) {
            cudaEventRecord(e0);
            cublasLtMatmul(lt, desc, &alpha, d_a, a_layout, d_b, b_layout,
                          &beta, d_c, c_layout, d_d, c_layout,
                          &heur[0].algo, d_workspace, workspace_size, 0);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = ops_per_gemm / (best/1000) / 1e12;
        printf("  %-12s: %.4f ms = %.0f TFLOPS\n", label, best, tflops);

        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(a_layout);
        cublasLtMatrixLayoutDestroy(b_layout);
        cublasLtMatrixLayoutDestroy(c_layout);
        cublasLtMatmulPreferenceDestroy(pref);
    };

    printf("# B300 GEMM benchmarks at N=K=M=%d\n\n", N);
    bench("FP16",    CUDA_R_16F,     CUDA_R_16F,     CUDA_R_16F,    CUBLAS_COMPUTE_32F);
    bench("BF16",    CUDA_R_16BF,    CUDA_R_16BF,    CUDA_R_16BF,   CUBLAS_COMPUTE_32F);
    bench("FP8 e4m3", CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF,  CUBLAS_COMPUTE_32F);
    // FP4 = e2m1
    bench("FP4 e2m1", CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16BF,  CUBLAS_COMPUTE_32F);

    cublasLtDestroy(lt);
    return 0;
}
