// Test power draw at peak tensor (cuBLAS FP8 GEMM)
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <chrono>
#include <thread>

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cudaStream_t s; cudaStreamCreate(&s);

    const int M = 8192;
    void *A, *B, *C;
    cudaMalloc(&A, (size_t)M*M);  // FP8
    cudaMalloc(&B, (size_t)M*M);
    cudaMalloc(&C, (size_t)M*M*2);  // BF16
    cudaMemset(A, 0x10, (size_t)M*M);
    cudaMemset(B, 0x10, (size_t)M*M);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t op_N = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &op_N, sizeof(op_N));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &op_N, sizeof(op_N));

    cublasLtMatrixLayout_t Ad, Bd, Cd;
    cublasLtMatrixLayoutCreate(&Ad, CUDA_R_8F_E4M3, M, M, M);
    cublasLtMatrixLayoutCreate(&Bd, CUDA_R_8F_E4M3, M, M, M);
    cublasLtMatrixLayoutCreate(&Cd, CUDA_R_16BF, M, M, M);

    float alpha = 1.0f, beta = 0.0f;
    size_t ws_sz = 32 * 1024 * 1024;
    void *ws; cudaMalloc(&ws, ws_sz);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz));

    cublasLtMatmulHeuristicResult_t heur;
    int returned;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref, 1, &heur, &returned);
    if (returned == 0) { printf("No algo!\n"); return 1; }

    // Warmup
    for (int i = 0; i < 3; i++)
        cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
    cudaDeviceSynchronize();

    printf("# B300 sustained FP8 GEMM 8192³ (4400 TFLOPS workload)\n");
    printf("# Runs ~10 sec, samples nvidia-smi during\n\n");

    // Run 10000 iters of FP8 GEMM (each ~0.25ms = 2.5 sec total per batch)
    auto run_batch = [&]{
        for (int i = 0; i < 10000; i++) {
            cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
        }
    };

    // Launch
    auto t0 = std::chrono::high_resolution_clock::now();
    run_batch();

    // Sample power during run
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    printf("After 0.5s of work:\n");
    system("nvidia-smi --query-gpu=power.draw,clocks.current.sm,temperature.gpu,utilization.gpu --format=csv,noheader | head -1");

    std::this_thread::sleep_for(std::chrono::milliseconds(2000));
    printf("After 2.5s:\n");
    system("nvidia-smi --query-gpu=power.draw,clocks.current.sm,temperature.gpu,utilization.gpu --format=csv,noheader | head -1");

    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    float total_s = std::chrono::duration<float>(t1-t0).count();

    printf("After completion (%.2f s):\n", total_s);
    system("nvidia-smi --query-gpu=power.draw,clocks.current.sm,temperature.gpu,utilization.gpu --format=csv,noheader | head -1");

    // Compute achieved TFLOPS
    double total_flops = 10000.0 * 2.0 * M * M * M;
    double tflops = total_flops / total_s / 1e12;
    printf("\nAchieved %.0f TFLOPS over %.2f sec\n", tflops, total_s);

    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ws);
    return 0;
}
