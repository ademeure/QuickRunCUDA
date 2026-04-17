// Measure each kernel's actual runtime when concurrent vs alone
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void ffma(float *out, int iters) {
    float a = threadIdx.x * 0.001f;
    float b = a + 0.001f, c = b + 0.001f, d = c + 0.001f;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f; b = b*1.0001f + 0.0001f;
        c = c*1.0001f + 0.0001f; d = d*1.0001f + 0.0001f;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cudaStream_t s_g, s_f;
    cudaStreamCreateWithFlags(&s_g, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_f, cudaStreamNonBlocking);

    int M = 8192;
    void *A, *B, *C;
    cudaMalloc(&A, (size_t)M*M); cudaMalloc(&B, (size_t)M*M); cudaMalloc(&C, (size_t)M*M*2);
    cudaMemset(A, 0, (size_t)M*M); cudaMemset(B, 0, (size_t)M*M);

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
    int ffma_iters = 100000;
    float alpha = 1.0f, beta = 0.0f;

    cudaEvent_t g0, g1, f0, f1;
    cudaEventCreate(&g0); cudaEventCreate(&g1);
    cudaEventCreate(&f0); cudaEventCreate(&f1);

    // Warmup
    for (int i = 0; i < 5; i++) {
        cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s_g);
        ffma<<<148, 256, 0, s_f>>>(d_out, ffma_iters);
    }
    cudaDeviceSynchronize();

    // Measure each separately during concurrent execution
    cudaEventRecord(g0, s_g);
    cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s_g);
    cudaEventRecord(g1, s_g);

    cudaEventRecord(f0, s_f);
    ffma<<<148, 256, 0, s_f>>>(d_out, ffma_iters);
    cudaEventRecord(f1, s_f);

    cudaDeviceSynchronize();

    float gemm_dur, ffma_dur;
    cudaEventElapsedTime(&gemm_dur, g0, g1);
    cudaEventElapsedTime(&ffma_dur, f0, f1);

    printf("# Concurrent run, each kernel timed via own stream's events:\n");
    printf("  GEMM duration (in concurrent run): %.2f ms\n", gemm_dur);
    printf("  FFMA duration (in concurrent run): %.2f ms\n", ffma_dur);

    // Now same kernels, alone
    cudaDeviceSynchronize();
    cudaEventRecord(g0, s_g);
    cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s_g);
    cudaEventRecord(g1, s_g);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&gemm_dur, g0, g1);

    cudaEventRecord(f0, s_f);
    ffma<<<148, 256, 0, s_f>>>(d_out, ffma_iters);
    cudaEventRecord(f1, s_f);
    cudaDeviceSynchronize();
    cudaEventElapsedTime(&ffma_dur, f0, f1);

    printf("\n# Alone:\n");
    printf("  GEMM alone: %.2f ms\n", gemm_dur);
    printf("  FFMA alone: %.2f ms\n", ffma_dur);

    return 0;
}
