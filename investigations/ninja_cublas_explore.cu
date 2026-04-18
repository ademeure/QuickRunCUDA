// Try ALL possible algo_ids (not just heuristic-returned) to find legacy mma.sync
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

int main() {
    int N = 8192;
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    void *d_a, *d_b, *d_c, *d_d, *d_ws;
    cudaMalloc(&d_a, (size_t)N*N*2); cudaMemset(d_a, 0x42, (size_t)N*N*2);
    cudaMalloc(&d_b, (size_t)N*N*2); cudaMemset(d_b, 0x42, (size_t)N*N*2);
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
    cublasLtMatrixLayoutCreate(&a, CUDA_R_16BF, N, N, N);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_16BF, N, N, N);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, N, N, N);

    float alpha = 1, beta = 0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    long ops = 2L * N * N * N;

    // Get list of all algo IDs supported
    int algo_ids[200]; int n_algos = 0;
    cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
        200, algo_ids, &n_algos);
    printf("# Total algo_ids supported for BF16 N=%d: %d\n", N, n_algos);
    for (int i = 0; i < n_algos; i++) printf(" %d", algo_ids[i]);
    printf("\n\n");

    // Try each
    for (int i = 0; i < n_algos; i++) {
        cublasLtMatmulAlgo_t algo;
        cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
            algo_ids[i], &algo);

        // Try default config
        cublasStatus_t st = cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b,
            &beta, d_c, c, d_d, c, &algo, d_ws, ws, 0);
        cudaError_t err = cudaDeviceSynchronize();
        if (st != CUBLAS_STATUS_SUCCESS || err != cudaSuccess) {
            printf("algo %3d: not supported (st=%d, err=%d)\n", algo_ids[i], (int)st, (int)err);
            cudaGetLastError();  // clear
            continue;
        }

        float best = 1e30f;
        for (int j = 0; j < 5; j++) {
            cudaEventRecord(e0);
            cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b,
                &beta, d_c, c, d_d, c, &algo, d_ws, ws, 0);
            cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = ops / (best/1000) / 1e12;
        printf("algo %3d: best %.4f ms = %.0f TFLOPS\n", algo_ids[i], best, tflops);
    }
    return 0;
}
