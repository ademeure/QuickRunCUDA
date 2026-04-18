// NVFP4 GEMM scaling with M (batch size) — find HBM↔compute crossover
//
// For inference decode: M=1, GEMV-like, HBM-bound
// For training/large-batch: M=8192, compute-bound
//
// Theoretical:
//   B300 NVFP4 peak: 10.8 PFLOPS (cuBLAS measured) or 15 PFLOPS spec
//   B300 HBM peak: 7.31 TB/s (read) or 7.0 effective for mixed
//   For weights N×K = 8192×8192 (NVFP4 = 4 bits each = 32 MB):
//     HBM read time = 32 MB / 7.0 TB/s = 4.6 us minimum
//     Compute time (M=1, 2*M*N*K ops): 2*1*8192*8192 / 10.8e15 = 12.4 ns (negligible)
//   Crossover: when compute time = HBM time
//     M_cross = HBM_size_bytes / (2 * N * K) * compute_TFLOPS / HBM_TB/s
//     For 8192x8192 NVFP4: 32 MB / (2*8192*8192) * 10.8e3 / 7 = 32M / 134M * 1543 ≈ 369
//     So crossover M ≈ 370
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

#define CHK(x) do { auto e = (x); if (e != cudaSuccess) { fprintf(stderr, "CUDA error %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); exit(1); } } while(0)

int main() {
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);
    int N = 8192, K = 8192;

    // Buffers
    void *d_a, *d_b, *d_c, *d_d, *d_a_sf, *d_b_sf, *d_ws;
    size_t a_max = 16384ULL * K / 2;       // M up to 16384
    size_t b_bytes = (size_t)K*N/2;
    size_t c_max = 16384ULL * N * 2;
    cudaMalloc(&d_a, a_max);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_max);
    cudaMalloc(&d_d, c_max);
    cudaMalloc(&d_a_sf, a_max / 8);  // VEC16 scale = 1 byte per 16 elems
    cudaMalloc(&d_b_sf, (size_t)K*N/16);
    size_t ws = 1024ull*1024*1024;
    cudaMalloc(&d_ws, ws);
    cudaMemset(d_a, 0x42, a_max);
    cudaMemset(d_b, 0x42, b_bytes);
    cudaMemset(d_a_sf, 0x40, a_max/8);
    cudaMemset(d_b_sf, 0x40, (size_t)K*N/16);

    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int M : {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192}) {
        cublasLtMatmulDesc_t desc;
        cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opT=CUBLAS_OP_T, opN=CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
        cublasLtMatmulMatrixScale_t sm = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm, sizeof(sm));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm, sizeof(sm));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_sf, sizeof(d_a_sf));
        cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_sf, sizeof(d_b_sf));

        cublasLtMatrixLayout_t la, lb, lc;
        cublasLtMatrixLayoutCreate(&la, CUDA_R_4F_E2M1, K, M, K);
        cublasLtMatrixLayoutCreate(&lb, CUDA_R_4F_E2M1, K, N, K);
        cublasLtMatrixLayoutCreate(&lc, CUDA_R_16BF, M, N, M);

        cublasLtMatmulPreference_t pref;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));

        cublasLtMatmulHeuristicResult_t heur[8];
        int nr = 0;
        cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 8, heur, &nr);
        if (nr == 0) { printf("M=%5d: no heur\n", M); continue; }

        float alpha = 1, beta = 0;
        for (int i = 0; i < 5; i++) cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
        cudaStreamSynchronize(s);

        // Run many to amortize launch overhead
        int n_iter = (M >= 1024) ? 20 : 100;
        cudaEventRecord(e0, s);
        for (int i = 0; i < n_iter; i++) cublasLtMatmul(lt, desc, &alpha, d_a, la, d_b, lb, &beta, d_c, lc, d_d, lc, &heur[0].algo, d_ws, ws, s);
        cudaEventRecord(e1, s);
        cudaStreamSynchronize(s);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        ms /= n_iter;

        // Bytes:
        //   A: M * K / 2 (NVFP4 is 4 bits)
        //   B: K * N / 2
        //   C (output BF16): M * N * 2
        size_t a_bytes_mem = (size_t)M * K / 2;
        size_t total_bytes = a_bytes_mem + b_bytes + (size_t)M * N * 2;
        long ops = 2L * M * N * K;
        double tflops = ops / (ms/1000.0) / 1e12;
        double effective_bw = total_bytes / (ms/1000.0) / 1e9;
        // Roofline:
        //   HBM-bound: total_bytes / 7.0e12 in seconds
        //   Compute-bound: ops / 10.8e15 in seconds
        double hbm_roof_us = total_bytes / 7.0e12 * 1e6;
        double compute_roof_us = ops / 10.8e15 * 1e6;
        printf("M=%5d  %.4f ms  TFLOPS=%6.1f  eff_BW=%5.0f GB/s  | HBM-roof=%.1f us  Compute-roof=%.1f us  measured=%.1f us\n",
               M, ms, tflops, effective_bw, hbm_roof_us, compute_roof_us, ms*1000);

        cublasLtMatrixLayoutDestroy(la);
        cublasLtMatrixLayoutDestroy(lb);
        cublasLtMatrixLayoutDestroy(lc);
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatmulDescDestroy(desc);
    }
    return 0;
}
