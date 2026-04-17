// cublas_alignment.cu
// Investigate the "4097 alignment cliff" claim for cuBLAS BF16 GEMM on B300 sm_103a / CUDA 13.2.
//
// Tests M=N=K and asymmetric shapes at powers-of-2 ± 1 to detect alignment-sensitive
// slow-paths in cuBLAS algorithm selection. Uses cublasGemmEx with CUDA_R_16BF input
// and CUDA_R_32F compute (CUBLAS_GEMM_DEFAULT and selected algo IDs).
//
// Also queries cublasLt heuristics to reveal which algorithm cuBLAS picks per size.
//
// Build: nvcc -arch=sm_103a -O3 -lcublas -lcublasLt -o cublas_alignment cublas_alignment.cu
// Run:   nvidia-smi -lgc 2032 && CUDA_VISIBLE_DEVICES=1 ./cublas_alignment

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <algorithm>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>

// ─────────────────────────────────────────────────────────
// Error macros
// ─────────────────────────────────────────────────────────
#define CUDA_CHECK(x) do {                                              \
    cudaError_t _e = (x);                                               \
    if (_e != cudaSuccess) {                                             \
        fprintf(stderr, "CUDA error %s:%d: %s\n",                       \
                __FILE__, __LINE__, cudaGetErrorString(_e));             \
        exit(1);                                                         \
    }                                                                    \
} while (0)

#define CUBLAS_CHECK(x) do {                                            \
    cublasStatus_t _s = (x);                                            \
    if (_s != CUBLAS_STATUS_SUCCESS) {                                   \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n",                     \
                __FILE__, __LINE__, (int)_s);                            \
        exit(1);                                                         \
    }                                                                    \
} while (0)

static const char* cublasStatusStr(cublasStatus_t s) {
    switch (s) {
        case CUBLAS_STATUS_SUCCESS:          return "SUCCESS";
        case CUBLAS_STATUS_NOT_INITIALIZED:  return "NOT_INITIALIZED";
        case CUBLAS_STATUS_ALLOC_FAILED:     return "ALLOC_FAILED";
        case CUBLAS_STATUS_INVALID_VALUE:    return "INVALID_VALUE";
        case CUBLAS_STATUS_ARCH_MISMATCH:    return "ARCH_MISMATCH";
        case CUBLAS_STATUS_MAPPING_ERROR:    return "MAPPING_ERROR";
        case CUBLAS_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case CUBLAS_STATUS_INTERNAL_ERROR:   return "INTERNAL_ERROR";
        case CUBLAS_STATUS_NOT_SUPPORTED:    return "NOT_SUPPORTED";
        default:                             return "UNKNOWN";
    }
}

// ─────────────────────────────────────────────────────────
// Timing helper: run N iterations, return median ms
// ─────────────────────────────────────────────────────────
static double timedGemmEx(
    cublasHandle_t handle,
    int M, int N, int K,
    const void* dA, const void* dB, void* dC,
    cudaStream_t stream,
    int nwarmup = 5,
    int nruns   = 50)
{
    float alpha = 1.0f, beta = 0.0f;

    // Warmup
    for (int i = 0; i < nwarmup; i++) {
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     M, N, K,
                     &alpha,
                     dA, CUDA_R_16BF, M,
                     dB, CUDA_R_16BF, K,
                     &beta,
                     dC, CUDA_R_32F,  M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    // Timed
    cudaEvent_t t0, t1;
    CUDA_CHECK(cudaEventCreate(&t0));
    CUDA_CHECK(cudaEventCreate(&t1));
    CUDA_CHECK(cudaEventRecord(t0, stream));
    for (int i = 0; i < nruns; i++) {
        cublasGemmEx(handle,
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     M, N, K,
                     &alpha,
                     dA, CUDA_R_16BF, M,
                     dB, CUDA_R_16BF, K,
                     &beta,
                     dC, CUDA_R_32F,  M,
                     CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    }
    CUDA_CHECK(cudaEventRecord(t1, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
    CUDA_CHECK(cudaEventDestroy(t0));
    CUDA_CHECK(cudaEventDestroy(t1));
    return (double)(ms / nruns);
}

// ─────────────────────────────────────────────────────────
// cublasLt heuristic query — returns algo index (id) and
// workspace size for top-1 algo, or -1 if none found.
// ─────────────────────────────────────────────────────────
struct AlgoInfo {
    int  algo_id;        // cublasLtMatmulAlgo_t inner id (extracted via attribute)
    int  tile_id;        // CUBLASLT_ALGO_CONFIG_TILE_ID
    int  stages_id;      // CUBLASLT_ALGO_CONFIG_STAGES_ID
    int  split_k;        // CUBLASLT_ALGO_CONFIG_SPLITK_NUM
    int  cta_swizzle;    // CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING
    int  num_algos;      // how many algos were returned
    size_t workspace_req;// workspace bytes required by algo[0]
};

static AlgoInfo queryLtAlgo(
    cublasLtHandle_t lt,
    int M, int N, int K,
    size_t workspaceBytes)
{
    AlgoInfo info = {-1, -1, -1, -1, -1, 0, 0};

    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasStatus_t st = cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    if (st != CUBLAS_STATUS_SUCCESS) return info;

    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, M, K, M);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F,  M, N, M);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspaceBytes, sizeof(workspaceBytes));

    const int MAX_ALGOS = 8;
    cublasLtMatmulHeuristicResult_t results[MAX_ALGOS] = {};
    int returnedAlgos = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, opDesc,
                                   layoutA, layoutB, layoutC, layoutC,
                                   pref, MAX_ALGOS, results, &returnedAlgos);
    info.num_algos = returnedAlgos;

    if (returnedAlgos > 0) {
        info.workspace_req = results[0].workspaceSize;

        // Extract algo attributes
        int val = 0; size_t sval = 0;
        cublasLtMatmulAlgoConfigGetAttribute(&results[0].algo,
            CUBLASLT_ALGO_CONFIG_ID, &val, sizeof(val), &sval);
        info.algo_id = val;

        cublasLtMatmulAlgoConfigGetAttribute(&results[0].algo,
            CUBLASLT_ALGO_CONFIG_TILE_ID, &val, sizeof(val), &sval);
        info.tile_id = val;

        cublasLtMatmulAlgoConfigGetAttribute(&results[0].algo,
            CUBLASLT_ALGO_CONFIG_STAGES_ID, &val, sizeof(val), &sval);
        info.stages_id = val;

        cublasLtMatmulAlgoConfigGetAttribute(&results[0].algo,
            CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &val, sizeof(val), &sval);
        info.split_k = val;

        cublasLtMatmulAlgoConfigGetAttribute(&results[0].algo,
            CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &val, sizeof(val), &sval);
        info.cta_swizzle = val;
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(opDesc);
    return info;
}

// ─────────────────────────────────────────────────────────
// cublasLt timed run using top-1 heuristic algo
// ─────────────────────────────────────────────────────────
static double timedLtGemm(
    cublasLtHandle_t lt,
    int M, int N, int K,
    const void* dA, const void* dB, void* dC,
    void* dWorkspace, size_t workspaceBytes,
    cudaStream_t stream,
    int nwarmup = 5,
    int nruns   = 50)
{
    cublasLtMatmulDesc_t opDesc = nullptr;
    cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    cublasOperation_t opN = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

    cublasLtMatrixLayout_t layoutA, layoutB, layoutC;
    cublasLtMatrixLayoutCreate(&layoutA, CUDA_R_16BF, M, K, M);
    cublasLtMatrixLayoutCreate(&layoutB, CUDA_R_16BF, K, N, K);
    cublasLtMatrixLayoutCreate(&layoutC, CUDA_R_32F,  M, N, M);

    cublasLtMatmulPreference_t pref = nullptr;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspaceBytes, sizeof(workspaceBytes));

    const int MAX_ALGOS = 1;
    cublasLtMatmulHeuristicResult_t results[1] = {};
    int returnedAlgos = 0;
    cublasLtMatmulAlgoGetHeuristic(lt, opDesc,
                                   layoutA, layoutB, layoutC, layoutC,
                                   pref, MAX_ALGOS, results, &returnedAlgos);

    double ms_per_run = -1.0;
    if (returnedAlgos > 0) {
        float alpha = 1.0f, beta = 0.0f;

        // Warmup
        for (int i = 0; i < nwarmup; i++) {
            cublasLtMatmul(lt, opDesc,
                           &alpha, dA, layoutA, dB, layoutB,
                           &beta,  dC, layoutC, dC, layoutC,
                           &results[0].algo, dWorkspace, workspaceBytes, stream);
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int i = 0; i < nruns; i++) {
            cublasLtMatmul(lt, opDesc,
                           &alpha, dA, layoutA, dB, layoutB,
                           &beta,  dC, layoutC, dC, layoutC,
                           &results[0].algo, dWorkspace, workspaceBytes, stream);
        }
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        ms_per_run = (double)(ms / nruns);
        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
    }

    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(opDesc);
    return ms_per_run;
}

// ─────────────────────────────────────────────────────────
// Main
// ─────────────────────────────────────────────────────────
int main(int argc, char** argv) {
    // Target GPU 1 (GPU 0 is busy with other work)
    int device = 1;
    if (argc > 1) device = atoi(argv[1]);
    CUDA_CHECK(cudaSetDevice(device));

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    int cudaVer = 0;
    cudaRuntimeGetVersion(&cudaVer);
    int cublasVer = 0;
    cublasGetVersion(nullptr, &cublasVer);

    printf("=== cuBLAS BF16 GEMM Alignment Cliff Investigation ===\n");
    printf("GPU:         %s  (SM %d.%d, device %d)\n", prop.name, prop.major, prop.minor, device);
    printf("CUDA:        %d.%d\n", cudaVer/1000, (cudaVer%1000)/10);
    printf("cuBLAS:      %d.%d.%d\n",
           cublasVer/100000, (cublasVer%100000)/1000, cublasVer%1000);
    printf("\n");

    // ── Allocate max-sized buffers (16385^2 × 2 bytes for BF16)
    // A: M×K BF16, B: K×N BF16, C: M×N FP32
    const int MAXDIM = 16385;
    const size_t MAX_BF16 = (size_t)MAXDIM * MAXDIM * 2;  // BF16
    const size_t MAX_FP32 = (size_t)MAXDIM * MAXDIM * 4;  // FP32

    void *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, MAX_BF16));
    CUDA_CHECK(cudaMalloc(&dB, MAX_BF16));
    CUDA_CHECK(cudaMalloc(&dC, MAX_FP32));
    CUDA_CHECK(cudaMemset(dA, 0, MAX_BF16));
    CUDA_CHECK(cudaMemset(dB, 0, MAX_BF16));
    CUDA_CHECK(cudaMemset(dC, 0, MAX_FP32));

    // Workspace: 256 MB (generous — let cuBLAS pick its best algo)
    const size_t WS_BYTES = 256ULL * 1024 * 1024;
    void* dWorkspace;
    CUDA_CHECK(cudaMalloc(&dWorkspace, WS_BYTES));
    CUDA_CHECK(cudaMemset(dWorkspace, 0, WS_BYTES));

    cublasHandle_t handle;
    CUBLAS_CHECK(cublasCreate(&handle));

    cublasLtHandle_t lt;
    CUBLAS_CHECK(cublasLtCreate(&lt));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    // ─────────────────────────────────────────────────────
    // Section 1: Square GEMM sweep
    // Sizes: ±1 around 1K, 2K, 4K, 4128, 4160, 8K, 16K
    // ─────────────────────────────────────────────────────
    printf("=== Section 1: Square GEMM (M=N=K) ─ BF16 in, FP32 out, CUBLAS_GEMM_DEFAULT ===\n");
    printf("%-7s  %10s  %8s  %8s  %6s  %6s  %6s  %s\n",
           "M=N=K", "TFLOPS(Lt)", "ms(Lt)", "ms(Ex)", "algoID", "tile", "stagesID", "splitK/swiz");

    struct SquareSize { int mnk; };
    std::vector<int> sq_sizes = {
        // 1K
        1023, 1024, 1025,
        // 2K
        2047, 2048, 2049,
        // 4K cluster
        4095, 4096, 4097, 4098,
        // 4K + pad
        4128, 4129, 4160, 4161,
        // 8K
        8191, 8192, 8193, 8224, 8256,
        // 16K
        16383, 16384, 16385
    };

    // Use fewer runs for large sizes to keep runtime manageable
    auto nruns_for = [](int m) -> int {
        if (m >= 16000) return 10;
        if (m >= 8000)  return 20;
        return 50;
    };

    for (int mnk : sq_sizes) {
        // Check buffer bounds
        if ((size_t)mnk * mnk * 2 > MAX_BF16 || (size_t)mnk * mnk * 4 > MAX_FP32) {
            printf("%7d  SKIPPED (too large for pre-allocated buffers)\n", mnk);
            continue;
        }

        AlgoInfo ai = queryLtAlgo(lt, mnk, mnk, mnk, WS_BYTES);

        int nr = nruns_for(mnk);
        double ms_lt = timedLtGemm(lt, mnk, mnk, mnk,
                                    dA, dB, dC, dWorkspace, WS_BYTES,
                                    stream, 3, nr);
        double ms_ex = timedGemmEx(handle, mnk, mnk, mnk,
                                    dA, dB, dC, stream, 3, nr);

        double flops = 2.0 * (double)mnk * mnk * mnk;
        double tflops_lt = (ms_lt > 0) ? flops / (ms_lt * 1e9) : -1;

        printf("%7d  %10.2f  %8.3f  %8.3f  %6d  %6d  %8d  %d/%d\n",
               mnk, tflops_lt, ms_lt, ms_ex,
               ai.algo_id, ai.tile_id, ai.stages_id,
               ai.split_k, ai.cta_swizzle);
    }

    // ─────────────────────────────────────────────────────
    // Section 2: Asymmetric GEMMs around the 4097 cliff
    // ─────────────────────────────────────────────────────
    printf("\n=== Section 2: Asymmetric shapes around M/N/K=4096/4097 ===\n");
    printf("%-5s  %-5s  %-5s  %10s  %8s  %6s  %6s  %s\n",
           "M", "N", "K", "TFLOPS(Lt)", "ms(Lt)", "algoID", "tile", "stages");

    struct AsymCase { int M, N, K; };
    std::vector<AsymCase> asym_cases = {
        // Baseline
        {4096, 4096, 4096},
        // One dim bumped
        {4097, 4096, 4096},
        {4096, 4097, 4096},
        {4096, 4096, 4097},
        // Two dims bumped
        {4097, 4097, 4096},
        {4097, 4096, 4097},
        {4096, 4097, 4097},
        // All three
        {4097, 4097, 4097},
        // Round up to 4128
        {4128, 4096, 4096},
        {4096, 4128, 4096},
        {4096, 4096, 4128},
        {4128, 4128, 4096},
        {4128, 4128, 4128},
        // 4160
        {4160, 4160, 4160},
        // 4096+64=4160 asymmetric
        {4160, 4096, 4096},
        // Batched style: M large, N/K aligned
        {8192, 4096, 4096},
        {4097, 8192, 4096},
    };

    for (auto& c : asym_cases) {
        size_t abytes = (size_t)c.M * c.K * 2;
        size_t bbytes = (size_t)c.K * c.N * 2;
        size_t cbytes = (size_t)c.M * c.N * 4;
        if (abytes > MAX_BF16 || bbytes > MAX_BF16 || cbytes > MAX_FP32) {
            printf("%5d  %5d  %5d  SKIPPED\n", c.M, c.N, c.K);
            continue;
        }

        AlgoInfo ai = queryLtAlgo(lt, c.M, c.N, c.K, WS_BYTES);

        double ms_lt = timedLtGemm(lt, c.M, c.N, c.K,
                                    dA, dB, dC, dWorkspace, WS_BYTES,
                                    stream, 3, 30);

        double flops = 2.0 * (double)c.M * c.N * c.K;
        double tflops_lt = (ms_lt > 0) ? flops / (ms_lt * 1e9) : -1;

        printf("%5d  %5d  %5d  %10.2f  %8.3f  %6d  %6d  %8d\n",
               c.M, c.N, c.K, tflops_lt, ms_lt,
               ai.algo_id, ai.tile_id, ai.stages_id);
    }

    // ─────────────────────────────────────────────────────
    // Section 3: Mitigation — try with larger workspace
    // and explicit algo enumeration for M=N=K=4097
    // ─────────────────────────────────────────────────────
    printf("\n=== Section 3: Mitigation test at M=N=K=4097 ===\n");
    printf("Testing different workspace sizes:\n");
    printf("%-15s  %10s  %8s  %6s\n", "Workspace", "TFLOPS", "ms", "algoID");

    size_t ws_sizes[] = {
        0,
        1ULL * 1024 * 1024,        //  1 MB
        8ULL * 1024 * 1024,        //  8 MB
        32ULL * 1024 * 1024,       // 32 MB
        128ULL * 1024 * 1024,      // 128 MB
        256ULL * 1024 * 1024,      // 256 MB
    };
    const char* ws_labels[] = {"0 (none)", "1 MB", "8 MB", "32 MB", "128 MB", "256 MB"};

    for (int wi = 0; wi < (int)(sizeof(ws_sizes)/sizeof(ws_sizes[0])); wi++) {
        size_t ws = std::min(ws_sizes[wi], WS_BYTES);  // cap at what we allocated
        AlgoInfo ai = queryLtAlgo(lt, 4097, 4097, 4097, ws);
        double ms_lt = timedLtGemm(lt, 4097, 4097, 4097,
                                    dA, dB, dC, dWorkspace, ws,
                                    stream, 3, 30);
        double flops = 2.0 * 4097.0 * 4097.0 * 4097.0;
        double tflops = (ms_lt > 0) ? flops / (ms_lt * 1e9) : -1;
        printf("%-15s  %10.2f  %8.3f  %6d (tile=%d stages=%d splitK=%d)\n",
               ws_labels[wi], tflops, ms_lt,
               ai.algo_id, ai.tile_id, ai.stages_id, ai.split_k);
    }

    // ─────────────────────────────────────────────────────
    // Section 4: Full algo enumeration for 4096 vs 4097
    // ─────────────────────────────────────────────────────
    printf("\n=== Section 4: Top-8 algos for M=N=K=4096 vs 4097 ===\n");

    for (int mnk : {4096, 4097}) {
        printf("\nM=N=K=%d:\n", mnk);
        printf("  %-4s  %-6s  %-8s  %-7s  %-8s  %s\n",
               "Rank", "AlgoID", "TileID", "StageID", "SplitK", "Workspace");

        cublasLtMatmulDesc_t opDesc = nullptr;
        cublasLtMatmulDescCreate(&opDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
        cublasOperation_t opN = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN));
        cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN));

        cublasLtMatrixLayout_t lA, lB, lC;
        cublasLtMatrixLayoutCreate(&lA, CUDA_R_16BF, mnk, mnk, mnk);
        cublasLtMatrixLayoutCreate(&lB, CUDA_R_16BF, mnk, mnk, mnk);
        cublasLtMatrixLayoutCreate(&lC, CUDA_R_32F,  mnk, mnk, mnk);

        cublasLtMatmulPreference_t pref = nullptr;
        cublasLtMatmulPreferenceCreate(&pref);
        cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                             &WS_BYTES, sizeof(WS_BYTES));

        const int MAX_ALGOS = 8;
        cublasLtMatmulHeuristicResult_t results[MAX_ALGOS] = {};
        int returnedAlgos = 0;
        cublasLtMatmulAlgoGetHeuristic(lt, opDesc,
                                       lA, lB, lC, lC,
                                       pref, MAX_ALGOS, results, &returnedAlgos);

        printf("  (returned %d algos)\n", returnedAlgos);

        for (int ri = 0; ri < returnedAlgos; ri++) {
            int algo_id = -1, tile_id = -1, stages_id = -1, split_k = -1, cta_swizzle = -1;
            size_t sval;
            cublasLtMatmulAlgoConfigGetAttribute(&results[ri].algo,
                CUBLASLT_ALGO_CONFIG_ID, &algo_id, sizeof(algo_id), &sval);
            cublasLtMatmulAlgoConfigGetAttribute(&results[ri].algo,
                CUBLASLT_ALGO_CONFIG_TILE_ID, &tile_id, sizeof(tile_id), &sval);
            cublasLtMatmulAlgoConfigGetAttribute(&results[ri].algo,
                CUBLASLT_ALGO_CONFIG_STAGES_ID, &stages_id, sizeof(stages_id), &sval);
            cublasLtMatmulAlgoConfigGetAttribute(&results[ri].algo,
                CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k), &sval);
            cublasLtMatmulAlgoConfigGetAttribute(&results[ri].algo,
                CUBLASLT_ALGO_CONFIG_CTA_SWIZZLING, &cta_swizzle, sizeof(cta_swizzle), &sval);

            printf("  %-4d  %-6d  %-8d  %-7d  %-8d  %zu bytes\n",
                   ri, algo_id, tile_id, stages_id, split_k, results[ri].workspaceSize);

            // Time this specific algo
            float alpha = 1.0f, beta = 0.0f;
            for (int w = 0; w < 3; w++) {
                cublasLtMatmul(lt, opDesc, &alpha, dA, lA, dB, lB,
                               &beta, dC, lC, dC, lC,
                               &results[ri].algo, dWorkspace, WS_BYTES, stream);
            }
            CUDA_CHECK(cudaStreamSynchronize(stream));

            cudaEvent_t t0, t1;
            CUDA_CHECK(cudaEventCreate(&t0));
            CUDA_CHECK(cudaEventCreate(&t1));
            CUDA_CHECK(cudaEventRecord(t0, stream));
            for (int rr = 0; rr < 20; rr++) {
                cublasLtMatmul(lt, opDesc, &alpha, dA, lA, dB, lB,
                               &beta, dC, lC, dC, lC,
                               &results[ri].algo, dWorkspace, WS_BYTES, stream);
            }
            CUDA_CHECK(cudaEventRecord(t1, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
            double ms_per = ms / 20.0;
            double flops = 2.0 * (double)mnk * mnk * mnk;
            double tflops = flops / (ms_per * 1e9);
            printf("        -> %.3f ms  %.2f TFLOPS\n", ms_per, tflops);
            CUDA_CHECK(cudaEventDestroy(t0));
            CUDA_CHECK(cudaEventDestroy(t1));
        }

        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(lA);
        cublasLtMatrixLayoutDestroy(lB);
        cublasLtMatrixLayoutDestroy(lC);
        cublasLtMatmulDescDestroy(opDesc);
    }

    // ─────────────────────────────────────────────────────
    // Section 5: Granular scan around M=N=K=4090–4105
    // to find exact cliff boundary
    // ─────────────────────────────────────────────────────
    printf("\n=== Section 5: Fine-grained scan M=N=K=4090..4112 ===\n");
    printf("%-7s  %10s  %8s  %6s  %6s\n",
           "M=N=K", "TFLOPS", "ms", "algoID", "tileID");

    for (int mnk = 4090; mnk <= 4112; mnk++) {
        AlgoInfo ai = queryLtAlgo(lt, mnk, mnk, mnk, WS_BYTES);
        double ms_lt = timedLtGemm(lt, mnk, mnk, mnk,
                                    dA, dB, dC, dWorkspace, WS_BYTES,
                                    stream, 3, 30);
        double flops = 2.0 * (double)mnk * mnk * mnk;
        double tflops = (ms_lt > 0) ? flops / (ms_lt * 1e9) : -1;
        printf("%7d  %10.2f  %8.3f  %6d  %6d\n",
               mnk, tflops, ms_lt, ai.algo_id, ai.tile_id);
    }

    // Cleanup
    cublasLtDestroy(lt);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dWorkspace);

    printf("\n=== Done ===\n");
    return 0;
}
