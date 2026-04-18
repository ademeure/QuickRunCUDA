// A4: Tensor-parallel GEMM split across 2 GPUs
//
// Split: A replicated on both GPUs, B split by columns (N axis)
// Each GPU computes M × (N/2) output via local GEMM
// Optional all-gather of output via NVLink
//
// Theoretical:
//   Single GPU 16384³ BF16:  2 × 16384³ ops / 2.2 PF = 4.0 ms
//   2-GPU TP:               (1/2 ops × 2.2 PF) + 0.5 × all-gather = 2.0 ms + 0.4 ms = 2.4 ms
//   Speedup: 1.66× ideal
//
//   Single GPU 32768³ BF16:  2 × 32768³ ops / 2.2 PF = 32 ms
//   2-GPU TP:               16 ms + (16K × 16K × 2 BF16 = 0.5 GB / 763 GB/s = 0.66 ms allgather)
//                         = 16.7 ms
//   Speedup: 1.92×
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <chrono>

void run_gemm(cublasLtHandle_t lt, int M, int N, int K, void *d_a, void *d_b, void *d_c, void *d_d, void *d_ws, size_t ws, cudaStream_t s, float *time_ms) {
    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatrixLayout_t la, lb, lc;
    cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&lb, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, M, N, M);
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, heur, &nr);
    if (nr == 0) { *time_ms = -1; return; }
    float alpha = 1, beta = 0;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
    cudaStreamSynchronize(s);
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0, s);
        cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
        cudaEventRecord(e1, s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    *time_ms = best;
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulPreferenceDestroy(pref);
}

int main(int argc, char**argv) {
    int M = (argc > 1) ? atoi(argv[1]) : 16384;
    int N = M, K = M;

    cudaSetDevice(0); cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1); cudaDeviceEnablePeerAccess(0, 0);

    // Allocate on both GPUs
    void *a0, *b0, *c0, *d0, *ws0;
    void *a1, *b1, *c1, *d1, *ws1;
    size_t ws_bytes = 1024ull*1024*1024;
    cudaSetDevice(0);
    cudaMalloc(&a0, (size_t)M*K*2);
    cudaMalloc(&b0, (size_t)K*N*2);  // full B, will use only half cols
    cudaMalloc(&c0, (size_t)M*N*2);
    cudaMalloc(&d0, (size_t)M*N*2);
    cudaMalloc(&ws0, ws_bytes);
    cudaMemset(a0, 0x3c, (size_t)M*K*2); cudaMemset(b0, 0x3c, (size_t)K*N*2);
    cudaSetDevice(1);
    cudaMalloc(&a1, (size_t)M*K*2);
    cudaMalloc(&b1, (size_t)K*N*2);
    cudaMalloc(&c1, (size_t)M*N*2);
    cudaMalloc(&d1, (size_t)M*N*2);
    cudaMalloc(&ws1, ws_bytes);
    cudaMemset(a1, 0x3c, (size_t)M*K*2); cudaMemset(b1, 0x3c, (size_t)K*N*2);

    cublasLtHandle_t lt0, lt1;
    cudaSetDevice(0); cublasLtCreate(&lt0);
    cudaSetDevice(1); cublasLtCreate(&lt1);

    cudaStream_t s0, s1;
    cudaSetDevice(0); cudaStreamCreate(&s0);
    cudaSetDevice(1); cudaStreamCreate(&s1);

    long ops_full = 2L * M * N * K;
    printf("# A4: Tensor-parallel GEMM split across 2× B300\n");
    printf("# Shape: M=N=K=%d BF16, total ops = %.1f TF\n\n", M, ops_full/1e12);

    // === Single GPU baseline ===
    float t_single;
    cudaSetDevice(0);
    run_gemm(lt0, M, N, K, a0, b0, c0, d0, ws0, ws_bytes, s0, &t_single);
    double tflops_single = ops_full / (t_single/1000.0) / 1e12;
    printf("  Single-GPU full GEMM:    %.3f ms  %.1f TFLOPS\n", t_single, tflops_single);

    // === 2-GPU TP: each does half N (no comms) ===
    int N_half = N / 2;
    long ops_half = 2L * M * N_half * K;
    float t_g0, t_g1;
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaSetDevice(0); run_gemm(lt0, M, N_half, K, a0, b0, c0, d0, ws0, ws_bytes, s0, &t_g0);
    cudaSetDevice(1); run_gemm(lt1, M, N_half, K, a1, b1, c1, d1, ws1, ws_bytes, s1, &t_g1);
    auto t1 = std::chrono::high_resolution_clock::now();
    double tflops_g0 = ops_half / (t_g0/1000.0) / 1e12;
    double tflops_g1 = ops_half / (t_g1/1000.0) / 1e12;
    printf("  GPU 0 half (no comms):   %.3f ms  %.1f TFLOPS\n", t_g0, tflops_g0);
    printf("  GPU 1 half (no comms):   %.3f ms  %.1f TFLOPS\n", t_g1, tflops_g1);
    double max_t = (t_g0 > t_g1) ? t_g0 : t_g1;
    printf("  Max(g0, g1) = parallel:  %.3f ms  %.1f TFLOPS aggregate\n",
           max_t, ops_full / (max_t/1000.0) / 1e12);

    // === Estimated TP with all-gather ===
    // All-gather: each GPU sends its half of C to peer (1 GB/2 = 0.5 GB at 763 GB/s = 0.66 ms for 32K)
    double allgather_ms = (double)M * N_half * 2 / 763e9 * 1000;
    double tp_total_ms = max_t + allgather_ms;
    printf("  TP + allgather (est):    %.3f ms (%.3f compute + %.3f comm)\n",
           tp_total_ms, max_t, allgather_ms);
    printf("  Speedup vs single-GPU:   %.2fx (ideal 2.00x)\n", t_single / tp_total_ms);
    printf("  Efficiency:              %.1f%%\n", t_single / tp_total_ms / 2 * 100);

    return 0;
}
