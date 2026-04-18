// Deeper power audit: track clock during test, try more configs
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
#include <cstdlib>

void run_test(const char *label, cudaDataType_t in_type, cudaDataType_t compute,
              cublasComputeType_t comp_kind, int elem_bytes) {
    int N = 8192;
    cudaSetDevice(0);
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    size_t mat_bytes = (size_t)N*N*elem_bytes;
    cudaMalloc(&d_a, mat_bytes);
    cudaMalloc(&d_b, mat_bytes);
    cudaMalloc(&d_c, (size_t)N*N*4);
    cudaMalloc(&d_d, (size_t)N*N*4);
    size_t ws = 256ull*1024*1024;
    cudaMalloc(&d_ws, ws);

    // Random init
    unsigned char *h = (unsigned char*)malloc(mat_bytes);
    srand(42);
    for (size_t i = 0; i < mat_bytes; i++) h[i] = rand() & 0xff;
    cudaMemcpy(d_a, h, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h, mat_bytes, cudaMemcpyHostToDevice);
    free(h);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, comp_kind, compute);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, in_type, N, N, N);
    cublasLtMatrixLayoutCreate(&b, in_type, N, N, N);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, N, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    if (st != CUBLAS_STATUS_SUCCESS || nr == 0) {
        printf("%s: NO HEUR (st=%d, nr=%d)\n", label, (int)st, nr);
        return;
    }

    cudaStream_t s; cudaStreamCreate(&s);
    float alpha=1.5f, beta=0.5f;  // non-trivial alpha/beta
    for (int i = 0; i < 3; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaError_t err = cudaStreamSynchronize(s);
    if (err != cudaSuccess) { printf("%s: warmup err %s\n", label, cudaGetErrorString(err)); return; }

    // Time one call
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0, s);
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long ops = 2L * N * N * N;
    double tflops = ops / (best/1000) / 1e12;

    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 16; i++) cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaGraph_t graph; cudaStreamEndCapture(s, &graph);
    cudaGraphExec_t exec; cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);

    std::atomic<bool> done{false};
    std::vector<unsigned> w, mhz;
    std::thread sampler([&]() {
        while (!done) {
            unsigned x;
            if (nvmlDeviceGetPowerUsage(dev, &x) == NVML_SUCCESS) w.push_back(x);
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &x) == NVML_SUCCESS) mhz.push_back(x);
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
    });
    int n_launches = 0;
    auto t_start = std::chrono::steady_clock::now();
    while (true) {
        cudaGraphLaunch(exec, s);
        n_launches++;
        if ((n_launches & 0x3) == 0) {
            cudaStreamSynchronize(s);
            auto el = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count();
            if (el >= 15.0) break;
        }
    }
    cudaStreamSynchronize(s);
    done = true; sampler.join();

    auto pmax = *std::max_element(w.begin(), w.end());
    auto pavg = (unsigned)(std::accumulate(w.begin(), w.end(), 0ull) / w.size());
    auto mhzmin = *std::min_element(mhz.begin(), mhz.end());
    auto mhzmax = *std::max_element(mhz.begin(), mhz.end());
    int n_at_2032 = std::count(mhz.begin(), mhz.end(), 2032u);
    int n_throttled = mhz.size() - n_at_2032;
    printf("%-20s: per-call %.1fTF | sustain avg=%uW max=%uW | clk %u-%u MHz | throttled %d/%zu samples\n",
           label, tflops, pavg/1000, pmax/1000, mhzmin, mhzmax, n_throttled, mhz.size());

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a); cublasLtMatrixLayoutDestroy(b); cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulPreferenceDestroy(pref);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d); cudaFree(d_ws);
    cublasLtDestroy(lt);
}

int main() {
    nvmlInit();
    printf("# Power audit v2: random data, all precisions, clock-throttle tracking\n\n");

    run_test("FP8 e4m3 ", CUDA_R_8F_E4M3, CUDA_R_32F, CUBLAS_COMPUTE_32F, 1);
    run_test("FP8 e5m2 ", CUDA_R_8F_E5M2, CUDA_R_32F, CUBLAS_COMPUTE_32F, 1);
    run_test("BF16     ", CUDA_R_16BF,    CUDA_R_32F, CUBLAS_COMPUTE_32F, 2);
    run_test("FP16     ", CUDA_R_16F,     CUDA_R_32F, CUBLAS_COMPUTE_32F, 2);
    run_test("TF32 (FP32 input)", CUDA_R_32F, CUDA_R_32F, CUBLAS_COMPUTE_32F_FAST_TF32, 4);

    nvmlShutdown();
    return 0;
}
