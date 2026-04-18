// Power audit: sustained cuBLAS at multiple precisions × data patterns
// Devil's advocate test: prior 983W result used 0x42 constant data
// (lowest possible bit-flip activity). Random data should burn more power.

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

void run_test(const char *label, cudaDataType_t in_type, int elem_bytes,
              const char *data_pattern, void (*init_data)(unsigned char*, size_t)) {
    int N = 8192;
    cudaSetDevice(0);
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    size_t mat_bytes = (size_t)N*N*elem_bytes;
    cudaMalloc(&d_a, mat_bytes);
    cudaMalloc(&d_b, mat_bytes);
    cudaMalloc(&d_c, (size_t)N*N*2);
    cudaMalloc(&d_d, (size_t)N*N*2);
    size_t ws = 256ull*1024*1024;
    cudaMalloc(&d_ws, ws);

    // Init data
    unsigned char *h = (unsigned char*)malloc(mat_bytes);
    init_data(h, mat_bytes);
    cudaMemcpy(d_a, h, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h, mat_bytes, cudaMemcpyHostToDevice);
    free(h);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
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
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);
    if (nr == 0) { printf("%s/%s: NO HEUR\n", label, data_pattern); return; }

    cudaStream_t s; cudaStreamCreate(&s);
    float alpha=1, beta=0;
    for (int i = 0; i < 3; i++) cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaStreamSynchronize(s);

    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 16; i++) cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s);
    cudaGraph_t graph; cudaStreamEndCapture(s, &graph);
    cudaGraphExec_t exec; cudaGraphInstantiate(&exec, graph, NULL, NULL, 0);

    std::atomic<bool> done{false};
    std::vector<unsigned> w;
    std::thread sampler([&]() {
        while (!done) {
            unsigned x;
            if (nvmlDeviceGetPowerUsage(dev, &x) == NVML_SUCCESS) w.push_back(x);
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
    printf("%-12s %-12s: avg=%uW, max=%uW, %u samples\n",
           label, data_pattern, pavg/1000, pmax/1000, (unsigned)w.size());

    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(a);
    cublasLtMatrixLayoutDestroy(b);
    cublasLtMatrixLayoutDestroy(c);
    cublasLtMatmulPreferenceDestroy(pref);
    cudaGraphExecDestroy(exec); cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_d); cudaFree(d_ws);
    cublasLtDestroy(lt);
}

void init_zero(unsigned char *h, size_t n) { memset(h, 0, n); }
void init_const42(unsigned char *h, size_t n) { memset(h, 0x42, n); }
void init_random(unsigned char *h, size_t n) { srand(42); for (size_t i = 0; i < n; i++) h[i] = rand() & 0xff; }
void init_alternating(unsigned char *h, size_t n) {
    for (size_t i = 0; i < n; i++) h[i] = (i & 1) ? 0xFF : 0x00;  // max bit transitions
}

int main() {
    nvmlInit();
    printf("# Sustained 15-sec cuBLAS power, varying precision × data pattern\n");
    printf("# Data: zero (no flips), const42 (fixed), random (uniform), alt (FF/00)\n\n");

    run_test("FP8 e4m3", CUDA_R_8F_E4M3, 1, "zero",     init_zero);
    run_test("FP8 e4m3", CUDA_R_8F_E4M3, 1, "const42",  init_const42);
    run_test("FP8 e4m3", CUDA_R_8F_E4M3, 1, "random",   init_random);
    run_test("FP8 e4m3", CUDA_R_8F_E4M3, 1, "alt(FF00)", init_alternating);

    run_test("FP8 e5m2", CUDA_R_8F_E5M2, 1, "zero",     init_zero);
    run_test("FP8 e5m2", CUDA_R_8F_E5M2, 1, "random",   init_random);

    run_test("BF16",     CUDA_R_16BF,    2, "zero",     init_zero);
    run_test("BF16",     CUDA_R_16BF,    2, "random",   init_random);

    run_test("FP16",     CUDA_R_16F,     2, "random",   init_random);

    nvmlShutdown();
    return 0;
}
