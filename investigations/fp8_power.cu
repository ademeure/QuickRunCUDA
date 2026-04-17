// FP8 cuBLAS GEMM at peak with power monitoring
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    cudaStream_t s; cudaStreamCreate(&s);

    nvmlInit_v2();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex_v2(0, &dev);

    int M = 8192;

    // Setup FP8 GEMM
    void *A, *B, *C;
    cudaMalloc(&A, (size_t)M*M);
    cudaMalloc(&B, (size_t)M*M);
    cudaMalloc(&C, (size_t)M*M*2);  // BF16 output

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
    if (returned == 0) { printf("No FP8 algo\n"); return 1; }

    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
    cudaStreamSynchronize(s);

    printf("# B300 FP8 cuBLAS sustained load with power monitoring\n");
    printf("# %-8s %-10s %-10s %-10s %-12s\n", "t_s", "Power_W", "Temp_C", "Clk_MHz", "TFLOPS");

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto t_start = std::chrono::high_resolution_clock::now();
    while (true) {
        cudaEventRecord(e0, s);
        for (int i = 0; i < 100; i++)  // ~50 ms batch
            cublasLtMatmul(lt, desc, &alpha, A, Ad, B, Bd, &beta, C, Cd, C, Cd, &heur.algo, ws, ws_sz, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);

        float ms; cudaEventElapsedTime(&ms, e0, e1);
        double tflops = 100.0 * 2.0 * M * M * M / (ms/1000) / 1e12;

        unsigned int pw, t_c, clk;
        nvmlDeviceGetPowerUsage(dev, &pw);
        nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t_c);
        nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &clk);

        auto t_now = std::chrono::high_resolution_clock::now();
        float t_s = std::chrono::duration<float>(t_now - t_start).count();
        printf("  %-8.1f %-10.1f %-10u %-10u %-12.0f\n", t_s, pw/1000.0, t_c, clk, tflops);
        if (t_s > 12) break;
    }

    nvmlShutdown();
    return 0;
}
