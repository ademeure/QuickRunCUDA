// Concurrent FP8 GEMM + FFMA - do tensor cores share with CUDA cores?
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void ffma_load(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    float b = a + 0.002f;
    float c = b + 0.003f;
    float d = c + 0.004f;
    for (int i = 0; i < iters; i++) {
        a = a*k1 + k2; b = b*k1 + k2; c = c*k1 + k2; d = d*k1 + k2;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cudaStream_t s_gemm, s_ffma;
    cudaStreamCreateWithFlags(&s_gemm, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_ffma, cudaStreamNonBlocking);

    int M = 8192;
    void *A, *B, *C;
    cudaMalloc(&A, (size_t)M*M);
    cudaMalloc(&B, (size_t)M*M);
    cudaMalloc(&C, (size_t)M*M*2);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t Ad, Bd, Cd;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, M, M, M);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, M, M, M);
    cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, M, M, M);
    size_t ws_sz = 64 * 1024 * 1024;
    void *ws; cudaMalloc(&ws, ws_sz);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz));
    cublasLtMatmulHeuristicResult_t heur;
    int returned;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref, 1, &heur, &returned);

    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));
    int ffma_iters = 100000;  // long FFMA
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < 3; i++) {
        cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s_gemm);
        ffma_load<<<148, 256, 0, s_ffma>>>(d_out, 1000, 1.0001f, 0.0001f);
    }
    cudaDeviceSynchronize();

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 concurrent FP8 GEMM + FFMA (do tensor cores share with CUDA cores?)\n\n");

    // FP8 GEMM alone
    float t_gemm = bench([&]{
        cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s_gemm);
    });
    printf("  FP8 GEMM alone:    %.2f ms\n", t_gemm);

    // FFMA alone
    float t_ffma = bench([&]{
        ffma_load<<<148, 256, 0, s_ffma>>>(d_out, ffma_iters, 1.0001f, 0.0001f);
    });
    printf("  FFMA alone:        %.2f ms\n", t_ffma);

    // Both concurrent (separate streams)
    float t_both = bench([&]{
        cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s_gemm);
        ffma_load<<<148, 256, 0, s_ffma>>>(d_out, ffma_iters, 1.0001f, 0.0001f);
    });
    printf("  Both concurrent:   %.2f ms (max of both = %.2f, sum = %.2f)\n",
           t_both, std::max(t_gemm, t_ffma), t_gemm + t_ffma);

    if (t_both <= 1.05f * std::max(t_gemm, t_ffma)) {
        printf("  → FULLY CONCURRENT (tensor and CUDA cores are SEPARATE units)\n");
    } else if (t_both >= 0.95f * (t_gemm + t_ffma)) {
        printf("  → SERIALIZED (no overlap)\n");
    } else {
        printf("  → PARTIAL OVERLAP\n");
    }

    return 0;
}
