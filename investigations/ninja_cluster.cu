// Try forcing 2sm cluster for FP4 to unlock 256x256x96 kernels
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

int main() {
    int M=8192, N=8192, K=12288;
    cudaSetDevice(0);
    cublasLtHandle_t lt; cublasLtCreate(&lt);

    size_t a_bytes = (size_t)M*K/2, b_bytes = (size_t)K*N/2, c_bytes = (size_t)M*N*2;
    void *d_a, *d_b, *d_c, *d_d, *d_a_scale, *d_b_scale, *d_bias, *d_ws;
    cudaMalloc(&d_a, a_bytes); cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes); cudaMalloc(&d_d, c_bytes);
    cudaMalloc(&d_a_scale, (size_t)M*K/16);
    cudaMalloc(&d_b_scale, (size_t)K*N/16);
    cudaMalloc(&d_bias, M * 2);
    size_t ws = 256ull*1024*1024; cudaMalloc(&d_ws, ws);
    unsigned char *h = (unsigned char*)malloc(b_bytes);
    srand(42);
    for (size_t i = 0; i < b_bytes; i++) h[i] = rand() & 0xff;
    cudaMemcpy(d_a, h, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_a_scale, h, (size_t)M*K/16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b_scale, h, (size_t)K*N/16, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h, M * 2, cudaMemcpyHostToDevice);
    free(h);

    cublasLtMatmulDesc_t desc;
    cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof(opT));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));
    cublasLtMatmulMatrixScale_t sm_ = CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &sm_, sizeof(sm_));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &sm_, sizeof(sm_));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &d_a_scale, sizeof(d_a_scale));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &d_b_scale, sizeof(d_b_scale));
    cublasLtEpilogue_t epi = CUBLASLT_EPILOGUE_GELU_BIAS;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_EPILOGUE, &epi, sizeof(epi));
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &d_bias, sizeof(d_bias));
    cudaDataType_t bias_dtype = CUDA_R_16BF;
    cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bias_dtype, sizeof(bias_dtype));

    cublasLtMatrixLayout_t a, b, c;
    cublasLtMatrixLayoutCreate(&a, CUDA_R_4F_E2M1, K, M, K);
    cublasLtMatrixLayoutCreate(&b, CUDA_R_4F_E2M1, K, N, K);
    cublasLtMatrixLayoutCreate(&c, CUDA_R_16BF, M, N, M);

    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur[1]; int nr;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 1, heur, &nr);

    int cur_cs = 0;
    cublasLtMatmulAlgoConfigGetAttribute(&heur[0].algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
        &cur_cs, sizeof(cur_cs), nullptr);
    printf("# Current heuristic cluster_shape = %d\n", cur_cs);
    // Try cluster shapes 0-20
    int cluster_shapes[20];
    for (int i = 0; i < 20; i++) cluster_shapes[i] = i;
    int n_cs = 20;

    cudaStream_t s; cudaStreamCreate(&s);
    float alpha=1, beta=0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    long ops = 2L * M * N * K;

    for (int ci = 0; ci < n_cs; ci++) {
        int cs = cluster_shapes[ci];
        cublasLtMatmulAlgo_t algo = heur[0].algo;
        cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_CLUSTER_SHAPE_ID,
            &cs, sizeof(cs));
        cublasLtMatmulHeuristicResult_t check;
        memset(&check, 0, sizeof(check));
        cublasStatus_t cst = cublasLtMatmulAlgoCheck(lt, desc, a, b, c, c, &algo, &check);
        if (cst != 0) { printf("  cluster %d: check fail (st=%d)\n", cs, (int)cst); continue; }

        cublasStatus_t st = cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta,
            d_c, c, d_d, c, &algo, d_ws, ws, s);
        cudaError_t err = cudaStreamSynchronize(s);
        if (st != 0 || err != cudaSuccess) {
            printf("  cluster %d: run fail (st=%d, err=%s)\n", cs, (int)st, cudaGetErrorString(err));
            cudaGetLastError();
            continue;
        }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0, s);
            cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b, &beta,
                d_c, c, d_d, c, &algo, d_ws, ws, s);
            cudaEventRecord(e1, s); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        double tflops = ops / (best/1000) / 1e12;
        printf("  cluster_shape %2d: %.4f ms = %.0f TFLOPS%s\n",
               cs, best, tflops, tflops > 9500 ? " <-- BREAKTHROUGH" : "");
    }
    return 0;
}
