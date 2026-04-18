// Free rein: data-dependent throughput across precisions
// I4 showed FP8 random data is 9-43% slower than zero. Does this generalize?
//
// Test: cuBLAS GEMM at BF16, FP16, FP8 with 4 data patterns:
//   - all zeros
//   - constant byte (0x42)
//   - random uniform
//   - random normal-ish (centered around 0, with some large values)

#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CHECK(x) do { auto e=(x); if (e!=cudaSuccess){printf("CUDA err %d\n",(int)e);exit(1);}} while(0)

int main() {
    int N = 8192;
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    void *d_a, *d_b, *d_c, *d_d, *d_workspace;
    CHECK(cudaMalloc(&d_a, (size_t)N*N*2));  // up to BF16
    CHECK(cudaMalloc(&d_b, (size_t)N*N*2));
    CHECK(cudaMalloc(&d_c, (size_t)N*N*2));
    CHECK(cudaMalloc(&d_d, (size_t)N*N*2));
    size_t ws = 256ull * 1024 * 1024;
    CHECK(cudaMalloc(&d_workspace, ws));

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto run_gemm = [&](cudaDataType_t in_type, cudaDataType_t out_type, int elem_bytes,
                        const char* prec_name) {
        cublasLtMatmulDesc_t desc;
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        cublasLtMatrixLayout_t a_layout, b_layout, c_layout;
        cublasLtMatrixLayoutCreate(&a_layout, in_type, N, N, N);
        cublasLtMatrixLayoutCreate(&b_layout, in_type, N, N, N);
        cublasLtMatrixLayoutCreate(&c_layout, out_type, N, N, N);

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

        cublasLtMatmulHeuristicResult_t heur[1];
        int nr;
        if (cublasLtMatmulAlgoGetHeuristic(lt, desc, a_layout, b_layout, c_layout, c_layout, pref, 1, heur, &nr) != CUBLAS_STATUS_SUCCESS || nr == 0) {
            printf("  %s: NO heuristic\n", prec_name);
            return;
        }

        printf("\n# %s GEMM (N=%d):\n", prec_name, N);

        const char *patterns[] = {"zero", "const42", "random", "normal-ish"};
        for (int p = 0; p < 4; p++) {
            // Init based on pattern
            unsigned char *h = (unsigned char*)malloc((size_t)N * N * elem_bytes);
            if (p == 0) memset(h, 0, (size_t)N*N*elem_bytes);
            else if (p == 1) memset(h, 0x42, (size_t)N*N*elem_bytes);
            else if (p == 2) {
                srand(42);
                for (size_t i = 0; i < (size_t)N*N*elem_bytes; i++) h[i] = rand() & 0xff;
            } else {  // normal-ish: mostly 0x3F-ish (small numbers in BF16) with some larger
                srand(43);
                if (in_type == CUDA_R_16BF) {
                    __nv_bfloat16 *bh = (__nv_bfloat16*)h;
                    for (size_t i = 0; i < (size_t)N*N; i++) {
                        float v = ((float)rand()/RAND_MAX) * 2 - 1;  // [-1, 1]
                        bh[i] = __float2bfloat16(v * v * v * 0.5f);  // small with occasional larger
                    }
                } else if (in_type == CUDA_R_16F) {
                    __half *hh = (__half*)h;
                    for (size_t i = 0; i < (size_t)N*N; i++) {
                        float v = ((float)rand()/RAND_MAX) * 2 - 1;
                        hh[i] = __float2half(v * v * v * 0.5f);
                    }
                } else { // FP8
                    for (size_t i = 0; i < (size_t)N*N; i++) {
                        float v = ((float)rand()/RAND_MAX) * 2 - 1;
                        unsigned char fp8 = (unsigned char)((v + 1) * 127);
                        h[i] = fp8;
                    }
                }
            }
            cudaMemcpy(d_a, h, (size_t)N*N*elem_bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(d_b, h, (size_t)N*N*elem_bytes, cudaMemcpyHostToDevice);
            free(h);

            float alpha = 1.0f, beta = 0.0f;
            for (int i = 0; i < 3; i++) {
                cublasLtMatmul(lt, desc, &alpha, d_a, a_layout, d_b, b_layout, &beta,
                              d_c, c_layout, d_d, c_layout, &heur[0].algo, d_workspace, ws, 0);
            }
            cudaDeviceSynchronize();
            float best = 1e30f;
            for (int i = 0; i < 10; i++) {
                cudaEventRecord(e0);
                cublasLtMatmul(lt, desc, &alpha, d_a, a_layout, d_b, b_layout, &beta,
                              d_c, c_layout, d_d, c_layout, &heur[0].algo, d_workspace, ws, 0);
                cudaEventRecord(e1); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) best = ms;
            }
            long ops = 2L * N * N * N;
            double tflops = ops / (best/1000) / 1e12;
            printf("  data=%-12s: %.4f ms = %.0f TFLOPS\n", patterns[p], best, tflops);
        }

        cublasLtMatmulDescDestroy(desc);
        cublasLtMatrixLayoutDestroy(a_layout);
        cublasLtMatrixLayoutDestroy(b_layout);
        cublasLtMatrixLayoutDestroy(c_layout);
        cublasLtMatmulPreferenceDestroy(pref);
    };

    run_gemm(CUDA_R_16F,    CUDA_R_16F,    2, "FP16");
    run_gemm(CUDA_R_16BF,   CUDA_R_16BF,   2, "BF16");
    run_gemm(CUDA_R_8F_E4M3, CUDA_R_16BF,  1, "FP8 e4m3");

    cublasLtDestroy(lt);
    return 0;
}
