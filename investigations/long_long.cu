// Run sustained FP8 GEMM for ~30 sec, check throttle behavior
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cudaStream_t s; cudaStreamCreate(&s);

    const int M = 8192;
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

    float alpha = 1.0f, beta = 0.0f;
    size_t ws_sz = 64 * 1024 * 1024;
    void *ws; cudaMalloc(&ws, ws_sz);

    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_sz, sizeof(ws_sz));
    cublasLtMatmulHeuristicResult_t heur;
    int returned;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, Ad, Bd, Cd, Cd, pref, 1, &heur, &returned);

    // Warmup
    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
    cudaDeviceSynchronize();

    printf("# B300 LONG sustained FP8 GEMM 8192³ - 30 sec test\n");
    printf("# %-6s %-12s %-15s %-50s\n", "batch", "elapsed_s", "TFLOPS", "nvidia-smi (W, MHz, °C)");

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    float baseline = 0;
    int batch = 0;
    auto wall_start = std::chrono::high_resolution_clock::now();

    while (true) {
        auto wall_now = std::chrono::high_resolution_clock::now();
        float wall_s = std::chrono::duration<float>(wall_now - wall_start).count();
        if (wall_s > 30.0) break;

        cudaEventRecord(e0, s);
        for (int i = 0; i < 200; i++)  // ~50 ms batch
            cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);

        float ms; cudaEventElapsedTime(&ms, e0, e1);
        float total_flops = 200.0 * 2.0 * M * M * M;
        float tflops = total_flops / (ms/1e3) / 1e12;
        if (batch == 0) baseline = tflops;

        if (batch == 0 || batch == 5 || batch % 50 == 0) {
            char clk_buf[128];
            FILE *p = popen("nvidia-smi --query-gpu=power.draw,clocks.current.sm,temperature.gpu --format=csv,noheader -i 0 | tr -d '\\n'", "r");
            fgets(clk_buf, sizeof(clk_buf), p); pclose(p);
            printf("  %-6d %-12.2f %-15.0f %-50s ratio=%.2f\n",
                   batch, wall_s, tflops, clk_buf, tflops / baseline);
        }
        batch++;
    }

    cudaEventDestroy(e0); cudaEventDestroy(e1);
    cudaFree(A); cudaFree(B); cudaFree(C); cudaFree(ws);

    printf("\n# Test complete. Final state:\n");
    system("nvidia-smi --query-gpu=power.draw,clocks.current.sm,temperature.gpu --format=csv,noheader -i 0");
    return 0;
}
