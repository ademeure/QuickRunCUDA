// Sustained FP8 cuBLAS — measure throughput + power over 30 seconds
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

int main() {
    int N = 8192;
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);

    cublasLtHandle_t lt; cublasLtCreate(&lt);
    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    cudaMalloc(&d_a, (size_t)N*N); cudaMemset(d_a, 0x42, (size_t)N*N);
    cudaMalloc(&d_b, (size_t)N*N); cudaMemset(d_b, 0x42, (size_t)N*N);
    cudaMalloc(&d_c, (size_t)N*N*2);
    cudaMalloc(&d_d, (size_t)N*N*2);
    size_t ws = 256ull * 1024 * 1024;
    cudaMalloc(&d_ws, ws);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_8F_E4M3, N, N, N);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_8F_E4M3, N, N, N);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, N, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    if (nr == 0) { printf("no heur\n"); return 1; }

    float alpha = 1, beta = 0;
    long ops = 2L * N * N * N;

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Background NVML sampler
    std::atomic<bool> done{false};
    std::vector<unsigned> samples_w, samples_t, samples_mhz;
    std::thread sampler([&]() {
        while (!done) {
            unsigned w, t, mhz;
            if (nvmlDeviceGetPowerUsage(dev, &w) == NVML_SUCCESS) samples_w.push_back(w);
            if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t) == NVML_SUCCESS) samples_t.push_back(t);
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz) == NVML_SUCCESS) samples_mhz.push_back(mhz);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // Warmup
    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, 0);
    cudaDeviceSynchronize();

    // Sustained run: queue 4096 launches up-front, then time 4096 more
    int batch = 4096;
    for (int i = 0; i < batch; i++)  // pre-fill queue
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, 0);
    cudaEventRecord(e0);
    for (int i = 0; i < batch; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, 0);
    cudaEventRecord(e1);
    cudaDeviceSynchronize();
    int n_iters = batch;
    done = true; sampler.join();

    float ms; cudaEventElapsedTime(&ms, e0, e1);
    double tflops_avg = (double)n_iters * ops * 1.0 / (ms/1000.0) / 1e12;

    auto pmin = *std::min_element(samples_w.begin(), samples_w.end());
    auto pmax = *std::max_element(samples_w.begin(), samples_w.end());
    auto pavg = std::accumulate(samples_w.begin(), samples_w.end(), 0u) / samples_w.size();
    auto tmin = *std::min_element(samples_t.begin(), samples_t.end());
    auto tmax = *std::max_element(samples_t.begin(), samples_t.end());
    auto mhzmin = *std::min_element(samples_mhz.begin(), samples_mhz.end());
    auto mhzmax = *std::max_element(samples_mhz.begin(), samples_mhz.end());

    printf("# Sustained FP8 GEMM N=%d for ~30 sec\n", N);
    printf("  Iterations: %d\n", n_iters);
    printf("  Wall: %.1f sec\n", ms/1000);
    printf("  Avg TFLOPS: %.1f\n", tflops_avg);
    printf("  Power: min=%u, avg=%u, max=%u W (across %zu samples)\n",
           pmin/1000, pavg/1000, pmax/1000, samples_w.size());
    printf("  Temp:  min=%u, max=%u C\n", tmin, tmax);
    printf("  Clock: min=%u, max=%u MHz\n", mhzmin, mhzmax);

    nvmlShutdown();
    return 0;
}
