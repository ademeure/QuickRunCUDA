// Try to push past 983W with FP8 cuBLAS + concurrent FFMA persistent
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

#define ITERS 5000
__launch_bounds__(256, 8) __global__ void ffma_burn(float *out, float a) {
    float r0=0.5f,r1=1.5f,r2=2.5f,r3=3.5f,r4=4.5f,r5=5.5f,r6=6.5f,r7=7.5f;
    float s0=8.5f,s1=9.5f,s2=10.5f,s3=11.5f,s4=12.5f,s5=13.5f,s6=14.5f,s7=15.5f;
    float t0=16.5f,t1=17.5f,t2=18.5f,t3=19.5f,t4=20.5f,t5=21.5f,t6=22.5f,t7=23.5f;
    float b = a + 1, c = a + 2;
    for (int i = 0; i < ITERS; i++) {
        r0=r0*a+b;r1=r1*a+c;r2=r2*a+b;r3=r3*a+c;r4=r4*a+b;r5=r5*a+c;r6=r6*a+b;r7=r7*a+c;
        s0=s0*b+a;s1=s1*b+c;s2=s2*b+a;s3=s3*b+c;s4=s4*b+a;s5=s5*b+c;s6=s6*b+a;s7=s7*b+c;
        t0=t0*c+a;t1=t1*c+b;t2=t2*c+a;t3=t3*c+b;t4=t4*c+a;t5=t5*c+b;t6=t6*c+a;t7=t7*c+b;
    }
    float sum = r0+r1+r2+r3+r4+r5+r6+r7+s0+s1+s2+s3+s4+s5+s6+s7+t0+t1+t2+t3+t4+t5+t6+t7;
    if (sum < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = sum;
}

int main() {
    int N = 8192;
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    cudaMalloc(&d_a, (size_t)N*N); cudaMemset(d_a, 0x42, (size_t)N*N);
    cudaMalloc(&d_b, (size_t)N*N); cudaMemset(d_b, 0x42, (size_t)N*N);
    cudaMalloc(&d_c, (size_t)N*N*2); cudaMalloc(&d_d, (size_t)N*N*2);
    size_t ws = 256ull*1024*1024;
    cudaMalloc(&d_ws, ws);
    float *d_out; cudaMalloc(&d_out, 1<<24);

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

    cudaStream_t s_gemm, s_ffma;
    cudaStreamCreate(&s_gemm); cudaStreamCreate(&s_ffma);
    float alpha=1, beta=0;

    for (int i = 0; i < 5; i++) cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s_gemm);
    cudaStreamSynchronize(s_gemm);

    // Capture both into graphs for sustained
    cudaStreamBeginCapture(s_gemm, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < 16; i++) cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta, d_c, c, d_d, c, &heur[0].algo, d_ws, ws, s_gemm);
    cudaGraph_t g_gemm; cudaStreamEndCapture(s_gemm, &g_gemm);
    cudaGraphExec_t e_gemm; cudaGraphInstantiate(&e_gemm, g_gemm, NULL, NULL, 0);

    std::atomic<bool> done{false};
    std::vector<unsigned> w;
    std::thread sampler([&]() {
        while (!done) {
            unsigned x;
            if (nvmlDeviceGetPowerUsage(dev, &x) == NVML_SUCCESS) w.push_back(x);
            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    });

    auto t_start = std::chrono::steady_clock::now();
    while (true) {
        cudaGraphLaunch(e_gemm, s_gemm);
        // Concurrently launch FFMA on different stream
        ffma_burn<<<148*4, 256, 0, s_ffma>>>(d_out, 1.5f);
        if ((int)(std::chrono::duration<double>(std::chrono::steady_clock::now() - t_start).count() * 10) > 100) break;  // ~10s
        cudaStreamSynchronize(s_gemm);
    }
    cudaStreamSynchronize(s_gemm);
    cudaStreamSynchronize(s_ffma);
    done = true; sampler.join();

    auto pmin = *std::min_element(w.begin(), w.end());
    auto pmax = *std::max_element(w.begin(), w.end());
    auto pavg = (unsigned)(std::accumulate(w.begin(), w.end(), 0ull) / w.size());
    printf("# FP8 cuBLAS + FFMA dual-stream concurrent\n");
    printf("  Power: min=%uW, avg=%uW, max=%uW\n", pmin/1000, pavg/1000, pmax/1000);
    nvmlShutdown();
    return 0;
}
