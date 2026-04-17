// cuSPARSE SpMM (sparse matrix dense matrix multiply)
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cusparseHandle_t h; cusparseCreate(&h);

    int M = 8192, N = 8192, K = 8192;
    int density_pct = 10;  // 10% non-zero
    int nnz = (long)M * K * density_pct / 100;

    printf("# B300 cuSPARSE SpMM (CSR sparse × dense, M=N=K=%d, %d%% sparsity)\n",
           M, density_pct);
    printf("# nnz = %d (%.1f MB CSR storage)\n\n",
           nnz, (long)nnz * 8 / 1024.0 / 1024.0);

    // Allocate sparse CSR matrix
    int *d_csr_offsets, *d_csr_cols;
    float *d_csr_vals, *d_B, *d_C;
    cudaMalloc(&d_csr_offsets, (M + 1) * sizeof(int));
    cudaMalloc(&d_csr_cols, nnz * sizeof(int));
    cudaMalloc(&d_csr_vals, nnz * sizeof(float));
    cudaMalloc(&d_B, (size_t)K * N * sizeof(float));
    cudaMalloc(&d_C, (size_t)M * N * sizeof(float));

    // Initialize sparse pattern (regular)
    int *h_offsets = new int[M + 1];
    int per_row = nnz / M;
    for (int i = 0; i <= M; i++) h_offsets[i] = i * per_row;
    cudaMemcpy(d_csr_offsets, h_offsets, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_csr_cols, 0, nnz * sizeof(int));
    cudaMemset(d_csr_vals, 0, nnz * sizeof(float));
    cudaMemset(d_B, 0, (size_t)K * N * sizeof(float));
    delete[] h_offsets;

    // SpMat descriptor
    cusparseSpMatDescr_t mat_a;
    cusparseCreateCsr(&mat_a, M, K, nnz, d_csr_offsets, d_csr_cols, d_csr_vals,
                      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

    cusparseDnMatDescr_t mat_b, mat_c;
    cusparseCreateDnMat(&mat_b, K, N, K, d_B, CUDA_R_32F, CUSPARSE_ORDER_COL);
    cusparseCreateDnMat(&mat_c, M, N, M, d_C, CUDA_R_32F, CUSPARSE_ORDER_COL);

    float alpha = 1.0f, beta = 0.0f;
    size_t ws_size = 0;
    cusparseSpMM_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                             &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, &ws_size);
    void *ws; cudaMalloc(&ws, ws_size);

    // Warmup
    for (int i = 0; i < 3; i++)
        cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, ws);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        cusparseSpMM(h, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_NON_TRANSPOSE,
                     &alpha, mat_a, mat_b, &beta, mat_c, CUDA_R_32F, CUSPARSE_SPMM_ALG_DEFAULT, ws);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    // Compare to dense GEMM (would be 2 * M * N * K)
    double dense_flops = 2.0 * M * N * K;
    double sparse_flops = 2.0 * nnz * N;  // SpMM does 2 ops per nnz × N cols
    printf("  SpMM time:    %.3f ms\n", best);
    printf("  Sparse FLOPs: %.0f G\n", sparse_flops/1e9);
    printf("  Dense equiv FLOPs: %.0f G\n", dense_flops/1e9);
    printf("  Effective TFLOPS: %.1f\n", sparse_flops / (best/1000) / 1e12);
    printf("  Workspace: %zu B\n", ws_size);

    return 0;
}
