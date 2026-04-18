// NINJA: cudaGraph-captured FP8 GEMM loop, sustained for 30 sec
// Tests: can B300 actually sustain near-peak power (962 W?) via cudaGraph?
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
    int N = 16384;
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

    cudaStream_t s; cudaStreamCreate(&s);

    // Warmup
    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaStreamSynchronize(s);

    // Capture a graph with 16 matmuls (so each launch is amortized over 16 calls)
    int per_graph = 16;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < per_graph; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaGraph_t graph;
    cudaStreamEndCapture(s, &graph);
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);

    // NVML sampler
    std::atomic<bool> done{false};
    std::vector<unsigned> w, t, mhz;
    std::thread sampler([&]() {
        while (!done) {
            unsigned x;
            if (nvmlDeviceGetPowerUsage(dev, &x) == NVML_SUCCESS) w.push_back(x);
            if (nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &x) == NVML_SUCCESS) t.push_back(x);
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &x) == NVML_SUCCESS) mhz.push_back(x);
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    });

    // Sustained run
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int n_iters = 0;
    cudaEventRecord(e0, s);
    auto t_start = std::chrono::steady_clock::now();
    while (true) {
        cudaGraphLaunch(exec, s);
        n_iters++;
        if ((n_iters & 0xF) == 0) {  // check periodically
            cudaStreamSynchronize(s);
            auto el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (el >= 30.0) break;
        }
    }
    cudaEventRecord(e1, s); cudaStreamSynchronize(s);
    done = true; sampler.join();

    float ms; cudaEventElapsedTime(&ms, e0, e1);
    long total_matmuls = (long)n_iters * per_graph;
    double tflops = total_matmuls * ops * 1.0 / (ms/1000.0) / 1e12;

    auto pmin = *std::min_element(w.begin(), w.end());
    auto pmax = *std::max_element(w.begin(), w.end());
    auto pavg = (unsigned)(std::accumulate(w.begin(), w.end(), 0ull) / w.size());
    auto tmax = *std::max_element(t.begin(), t.end());
    auto mhzmin = *std::min_element(mhz.begin(), mhz.end());
    auto mhzmax = *std::max_element(mhz.begin(), mhz.end());

    printf("# Sustained FP8 GEMM via cudaGraph (%d matmul/graph, ~30 sec)\n", per_graph);
    printf("  Graph launches: %d\n", n_iters);
    printf("  Total matmuls: %ld\n", total_matmuls);
    printf("  Wall: %.1f sec\n", ms/1000);
    printf("  Avg TFLOPS: %.1f\n", tflops);
    printf("  Power: min=%u, avg=%u, max=%u W\n", pmin/1000, pavg/1000, pmax/1000);
    printf("  Temp:  max=%u C\n", tmax);
    printf("  Clock: min=%u, max=%u MHz\n", mhzmin, mhzmax);

    nvmlShutdown();
    return 0;
}
