// Full NVFP4 audit: N-scaling + sustained power + random data
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <nvml.h>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cstdlib>

void measure(int N, bool sustained) {
    cudaSetDevice(0);
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    size_t mat_bytes = (size_t)N*N/2;  // FP4 = 0.5 byte/elem
    size_t scale_bytes = (size_t)N*N/16;  // 1 ue4m3 per 16 elems
    void *d_a, *d_b, *d_c, *d_d, *d_a_scale, *d_b_scale, *d_ws;
    cudaMalloc(&d_a, mat_bytes); cudaMalloc(&d_b, mat_bytes);
    cudaMalloc(&d_c, (size_t)N*N*2); cudaMalloc(&d_d, (size_t)N*N*2);
    cudaMalloc(&d_a_scale, scale_bytes); cudaMalloc(&d_b_scale, scale_bytes);
    size_t ws = 256ull*1024*1024; cudaMalloc(&d_ws, ws);

    // RANDOM data per pitfall #10
    unsigned char *h = (unsigned char*)malloc(mat_bytes);
    srand(42);
    for (size_t i = 0; i < mat_bytes; i++) h[i] = rand() & 0xff;
    cudaMemcpy(d_a, h, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h, mat_bytes, cudaMemcpyHostToDevice);
    free(h);
    h = (unsigned char*)malloc(scale_bytes);
    for (size_t i = 0; i < scale_bytes; i++) h[i] = 0x40 + (rand() & 0x07);  // small ue4m3 scales
    cudaMemcpy(d_a_scale, h, scale_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_scale, h, scale_bytes, cudaMemcpyHostToDevice);
    free(h);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatmulMatrixScale_t scale_mode = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &scale_mode, sizeof(scale_mode));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale));
    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, N, N, N);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, N, N, N);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, N, N);
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasStatus_t st = cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    if (st != 0 || nr == 0) { printf("N=%d: NO HEUR\n", N); return; }

    cudaStream_t s; cudaStreamCreate(&s);
    float alpha=1, beta=0;
    for (int i = 0; i < 5; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaStreamSynchronize(s);

    long ops = 2L * N * N * N;

    // Per-call timing (no graph)
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0, s);
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
        cudaEventRecord(e1, s); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    double per_call_tflops = ops / (best/1000) / 1e12;

    if (!sustained) {
        printf("NVFP4 N=%5d: per-call %.4f ms = %.0f TFLOPS = %.1f%% of 10000 spec\n",
               N, best, per_call_tflops, per_call_tflops/10000*100);
        return;
    }

    // Sustained via cudaGraph
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 16; i++)
        cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaGraph_t graph; cudaStreamEndCapture(s, &graph);
    cudaGraphExec_t exec; cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);

    // Sample power via field 186 (true instantaneous)
    std::atomic<bool> done{false};
    std::vector<unsigned> w_avg, w_inst, mhz;
    nvmlFieldValue_t fv;
    memset(&fv, 0, sizeof(fv));
    fv.fieldId = NVML_FI_DEV_POWER_INSTANT;
    std::thread sampler([&]() {
        while (!done) {
            unsigned x;
            if (nvmlDeviceGetPowerUsage(dev, &x) == NVML_SUCCESS) w_avg.push_back(x);
            if (nvmlDeviceGetFieldValues(dev, 1, &fv) == NVML_SUCCESS && fv.nvmlReturn == NVML_SUCCESS)
                w_inst.push_back(fv.value.uiVal);
            if (nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &x) == NVML_SUCCESS) mhz.push_back(x);
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
        }
    });

    cudaEventRecord(e0, s);
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
    cudaEventRecord(e1, s); cudaStreamSynchronize(s);
    done = true; sampler.join();

    float ms; cudaEventElapsedTime(&ms, e0, e1);
    long total_matmuls = (long)n_launches * 16;
    double tflops_sus = total_matmuls * ops * 1.0 / (ms/1000) / 1e12;

    auto p_avg_max = *std::max_element(w_avg.begin(), w_avg.end());
    auto p_avg_mean = (unsigned)(std::accumulate(w_avg.begin(), w_avg.end(), 0ull) / w_avg.size());
    auto p_inst_max = *std::max_element(w_inst.begin(), w_inst.end());
    auto mhzmin = *std::min_element(mhz.begin(), mhz.end());
    int n_at_2032 = std::count(mhz.begin(), mhz.end(), 2032u);

    printf("NVFP4 N=%5d sustained: per-call %.0f TFLOPS, sustained %.0f TFLOPS\n",
           N, per_call_tflops, tflops_sus);
    printf("  Power: avg(smoothed)=%uW max=%uW, INST_PEAK=%uW\n",
           p_avg_mean/1000, p_avg_max/1000, p_inst_max/1000);
    printf("  Clock: min=%u MHz, %d/%zu samples at 2032 (%.0f%% throttled)\n",
           mhzmin, (int)mhz.size() - n_at_2032, mhz.size(),
           (mhz.size() - n_at_2032) * 100.0 / mhz.size());
}

int main() {
    nvmlInit();
    printf("# NVFP4 audit (random data, field 186, with throttle tracking)\n\n");
    // N-sweep per-call
    for (int N : {2048, 4096, 8192, 12288, 16384, 24576, 32768}) measure(N, false);
    printf("\n# Sustained at peak N values:\n");
    measure(16384, true);
    measure(8192, true);
    nvmlShutdown();
    return 0;
}
