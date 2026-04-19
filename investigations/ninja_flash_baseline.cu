// A1: FlashAttention baseline via cuBLAS (no fused softmax)
//
// Pattern (per attention head):
//   P = Q × K^T / sqrt(D)         [T × T]
//   P = softmax(P) along last dim  [T × T]
//   O = P × V                      [T × D]
//
// THEORETICAL:
//   Per head: 2 × T × D × T (Q*K^T) + 2 × T × T × D (P*V) = 4 × T² × D
//   For B*H=128, T=2048, D=128:
//     ops = 128 × 4 × 2048² × 128 = 274 GFLOPS — way too small
//   Better: T=4096, B*H=32:
//     ops = 32 × 4 × 4096² × 128 = 274 GFLOPS still small
//   T=8192, B*H=64:
//     ops = 64 × 4 × 8192² × 128 = 2.2 TFLOPS, time at 2.2PF = 1 ms
//
// FlashAttention SoL ≤ cuBLAS GEMM SoL for the two matmuls.
// Real FlashAttention saves L2/HBM traffic by tiling — wins when memory-bound.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdio>

void run_gemm(cublasLtHandle_t lt, int M, int N, int K, void *d_a, void *d_b, void *d_c, void *d_d, void *d_ws, size_t ws, cudaStream_t s, float *time_ms, int n_iter = 5) {
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
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
    cudaStreamSynchronize(s);
    float best = 1e30f;
    cudaEventRecord(e0, s);
    for (int i = 0; i < n_iter; i++) cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
    cudaEventRecord(e1, s); cudaStreamSynchronize(s);
    cudaEventElapsedTime(&best, e0, e1);
    *time_ms = best / n_iter;
    cublasLtMatmulDescDestroy(desc);
    cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb); cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulPreferenceDestroy(pref);
}

// Simple row-wise softmax for [T x T] BF16
__launch_bounds__(256, 4) __global__ void k_softmax(__nv_bfloat16 *p, int T) {
    int row = blockIdx.x;
    extern __shared__ float smem[];
    // Pass 1: max
    float local_max = -1e9;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float v = __bfloat162float(p[row * T + j]);
        if (v > local_max) local_max = v;
    }
    smem[threadIdx.x] = local_max;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] = fmaxf(smem[threadIdx.x], smem[threadIdx.x+s]);
        __syncthreads();
    }
    float mx = smem[0];
    // Pass 2: sum exp
    float local_sum = 0;
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        local_sum += expf(__bfloat162float(p[row * T + j]) - mx);
    }
    smem[threadIdx.x] = local_sum;
    __syncthreads();
    for (int s = blockDim.x/2; s > 0; s >>= 1) {
        if (threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x+s];
        __syncthreads();
    }
    float sum = smem[0];
    float inv_sum = 1.0f / sum;
    // Pass 3: write
    for (int j = threadIdx.x; j < T; j += blockDim.x) {
        float v = expf(__bfloat162float(p[row * T + j]) - mx) * inv_sum;
        p[row * T + j] = __float2bfloat16(v);
    }
}

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    int T = 4096, D = 128, BH = 32;
    long ops_per_head = 4L * T * T * D;
    long total_ops = (long)BH * ops_per_head;

    void *Q, *K, *V, *P, *O, *ws_buf;
    cudaMalloc(&Q, (size_t)BH * T * D * 2);
    cudaMalloc(&K, (size_t)BH * T * D * 2);
    cudaMalloc(&V, (size_t)BH * T * D * 2);
    cudaMalloc(&P, (size_t)BH * T * T * 2);
    cudaMalloc(&O, (size_t)BH * T * D * 2);
    size_t ws = 1024ull*1024*1024;
    cudaMalloc(&ws_buf, ws);
    cudaMemset(Q, 0x3c, (size_t)BH * T * D * 2);
    cudaMemset(K, 0x3c, (size_t)BH * T * D * 2);
    cudaMemset(V, 0x3c, (size_t)BH * T * D * 2);

    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    printf("# A1: FlashAttention baseline via cuBLAS + softmax kernel\n");
    printf("# Shape: T=%d, D=%d, BH=%d → total ops = %.1f GFLOPS\n", T, D, BH, total_ops/1e9);

    // --- Single head GEMM 1: Q × K^T → P  (M=N=T=8192, K=D=128) ---
    float t_qk;
    run_gemm(lt, T, T, D, Q, K, P, P, ws_buf, ws, s, &t_qk);
    long ops_qk = 2L * T * T * D;
    double tflops_qk = ops_qk / (t_qk/1000.0) / 1e12;
    printf("\n  Single head Q×K^T (M=N=%d K=%d): %.4f ms  %.1f TFLOPS  (%.1f%% spec)\n",
           T, D, t_qk, tflops_qk, tflops_qk/2500*100);

    // --- Softmax row-wise on P (T × T) ---
    for (int i = 0; i < 3; i++) k_softmax<<<T, 256, 256*4, s>>>((__nv_bfloat16*)P, T);
    cudaStreamSynchronize(s);
    float t_softmax = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0, s);
        k_softmax<<<T, 256, 256*4, s>>>((__nv_bfloat16*)P, T);
        cudaEventRecord(e1, s); cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < t_softmax) t_softmax = ms;
    }
    printf("  Single head softmax (T×T):       %.4f ms\n", t_softmax);

    // --- GEMM 2: P × V → O (M=T, N=D, K=T) ---
    float t_pv;
    run_gemm(lt, T, D, T, P, V, O, O, ws_buf, ws, s, &t_pv);
    long ops_pv = 2L * T * D * T;
    double tflops_pv = ops_pv / (t_pv/1000.0) / 1e12;
    printf("  Single head P×V (M=%d N=%d K=%d): %.4f ms  %.1f TFLOPS  (%.1f%% spec)\n",
           T, D, T, t_pv, tflops_pv, tflops_pv/2500*100);

    // --- Total per single head ---
    double t_one = t_qk + t_softmax + t_pv;
    printf("\n  Single head total: %.3f ms  → %.1f TFLOPS effective\n",
           t_one, ops_per_head / (t_one/1000.0) / 1e12);

    // --- Multi-head end-to-end (BH heads sequential) ---
    cudaStreamSynchronize(s);
    cudaEventRecord(e0, s);
    for (int b = 0; b < BH; b++) {
        run_gemm(lt, T, T, D, (char*)Q + b*T*D*2, (char*)K + b*T*D*2, (char*)P + b*T*T*2, (char*)P + b*T*T*2, ws_buf, ws, s, &t_qk, 1);
        k_softmax<<<T, 256, 256*4, s>>>((__nv_bfloat16*)((char*)P + b*T*T*2), T);
        run_gemm(lt, T, D, T, (char*)P + b*T*T*2, (char*)V + b*T*D*2, (char*)O + b*T*D*2, (char*)O + b*T*D*2, ws_buf, ws, s, &t_pv, 1);
    }
    cudaEventRecord(e1, s); cudaStreamSynchronize(s);
    float t_total; cudaEventElapsedTime(&t_total, e0, e1);
    double tflops_total = (double)total_ops / (t_total/1000.0) / 1e12;
    printf("\n  All %d heads end-to-end: %.3f ms  → %.1f TFLOPS  (%.1f%% spec)\n",
           BH, t_total, tflops_total, tflops_total/2500*100);

    return 0;
}
