// Force-iterate ALL algos + tile_ids for NVFP4 with bias+gelu to find K=96 path
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <vector>

int main() {
    int M=8192, N=8192, K=12288;  // K=96*128
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

    // First get the heuristic's algo as a baseline
    cublasLtMatmulPreference_t pref; cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof(ws));
    cublasLtMatmulHeuristicResult_t heur_arr[32]; int heur_count = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, desc, a, b, c, c, pref, 32, heur_arr, &heur_count);
    printf("# Heuristic returned %d algos\n", heur_count);
    for (int i = 0; i < heur_count; i++) {
        int aid=0, tid=0;
        cublasLtMatmulAlgoConfigGetAttribute(&heur_arr[i].algo, CUBLASLT_ALGO_CONFIG_ID, &aid, sizeof(aid), nullptr);
        cublasLtMatmulAlgoConfigGetAttribute(&heur_arr[i].algo, CUBLASLT_ALGO_CONFIG_TILE_ID, &tid, sizeof(tid), nullptr);
        printf("   heur[%d] algo=%d tile=%d ws=%zu\n", i, aid, tid, heur_arr[i].workspaceSize);
    }
    printf("\n");

    // Algo IDs from get_ids
    int algo_ids[300]; int n_algos = 0;
    cublasLtMatmulAlgoGetIds(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F,
        CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16BF, CUDA_R_16BF,
        300, algo_ids, &n_algos);
    printf("# %d algo_ids available for FP4 e2m1\n", n_algos);
    for (int i = 0; i < n_algos; i++) printf(" %d", algo_ids[i]); printf("\n\n");

    cudaStream_t s; cudaStreamCreate(&s);
    float alpha=1, beta=0;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    long ops = 2L * M * N * K;
    double best_tflops = 0;
    int best_algo = -1, best_tile = -1;

    for (int ai = 0; ai < n_algos; ai++) {
        int algo_id = algo_ids[ai];

        // Get supported tiles for this algo
        cublasLtMatmulAlgo_t algo;
        cublasLtMatmulAlgoInit(lt, CUBLAS_COMPUTE_32F, CUDA_R_32F,
            CUDA_R_4F_E2M1, CUDA_R_4F_E2M1, CUDA_R_16BF, CUDA_R_16BF, algo_id, &algo);

        // First get size needed
        size_t tiles_size = 0;
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS,
            nullptr, 0, &tiles_size);
        int n_tiles = tiles_size / sizeof(int);
        if (n_tiles <= 0 || n_tiles > 1000) n_tiles = 0;
        std::vector<int> tiles(n_tiles > 0 ? n_tiles : 300);
        cublasLtMatmulAlgoCapGetAttribute(&algo, CUBLASLT_ALGO_CAP_TILE_IDS,
            tiles.data(), tiles.size() * sizeof(int), &tiles_size);
        n_tiles = tiles_size / sizeof(int);
        printf("algo %d: %d tiles\n", algo_id, n_tiles);

        int n_succeeded = 0;
        // Use heur[0]'s fully-configured algo as template, change only tile_id
        cublasLtMatmulAlgo_t template_algo;
        if (heur_count > 0) template_algo = heur_arr[0].algo;
        for (int ti = 0; ti < n_tiles; ti++) {
            int tile = tiles[ti];
            algo = template_algo;  // start from heuristic's config
            cublasLtMatmulAlgoConfigSetAttribute(&algo, CUBLASLT_ALGO_CONFIG_TILE_ID,
                &tile, sizeof(tile));

            // Check if supported via heuristic_check
            cublasLtMatmulHeuristicResult_t check;
            memset(&check, 0, sizeof(check));
            cublasStatus_t cst = cublasLtMatmulAlgoCheck(lt, desc, a, b, c, c, &algo, &check);
            if (cst != 0) continue;

            // Try a run
            cublasStatus_t st = cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b,
                &beta, d_c, c, d_d, c, &algo, d_ws, ws, s);
            cudaError_t err = cudaStreamSynchronize(s);
            if (st != 0 || err != cudaSuccess) { cudaGetLastError(); continue; }

            float best = 1e30f;
            for (int i = 0; i < 5; i++) {
                cudaEventRecord(e0, s);
                cublasLtMatmul(lt, desc, &alpha, d_a, a, d_b, b,
                    &beta, d_c, c, d_d, c, &algo, d_ws, ws, s);
                cudaEventRecord(e1, s); cudaEventSynchronize(e1);
                float ms; cudaEventElapsedTime(&ms, e0, e1);
                if (ms < best) best = ms;
            }
            double tflops = ops / (best/1000) / 1e12;
            n_succeeded++;
            printf("  algo %3d tile %4d: %.4f ms = %.0f TFLOPS%s\n",
                   algo_id, tile, best, tflops,
                   tflops > 9500 ? " <-- BREAKTHROUGH!" : "");
            if (tflops > best_tflops) {
                best_tflops = tflops;
                best_algo = algo_id;
                best_tile = tile;
            }
        }
        printf("  %d/%d tiles succeeded for algo %d\n", n_succeeded, n_tiles, algo_id);
    }

    printf("\n## BEST: algo %d tile %d = %.0f TFLOPS = %.1f%% of 10000 spec\n",
           best_algo, best_tile, best_tflops, best_tflops/10000*100);
    return 0;
}
