// cublas_fp8_check.cu
// Definitive test: does cublasLt FP8 GEMM work on B300 sm_103a with CUDA 13.2?
//
// Tests E4M3, E5M2 inputs against BF16/F32/E4M3 outputs.
// Also runs FP16 and BF16 as sanity-check controls.
// Measures TFLOPS where supported.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cassert>
#include <vector>
#include <string>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cublas_v2.h>

// ──────────────────────────────────────────────────────────────────────────────
// Error-checking macros
// ──────────────────────────────────────────────────────────────────────────────
#define CUDA_CHECK(x)                                                    \
  do {                                                                   \
    cudaError_t _e = (x);                                               \
    if (_e != cudaSuccess) {                                             \
      fprintf(stderr, "CUDA error %s:%d: %s\n",                        \
              __FILE__, __LINE__, cudaGetErrorString(_e));               \
      exit(1);                                                           \
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
        case CUBLAS_STATUS_LICENSE_ERROR:    return "LICENSE_ERROR";
        default: return "UNKNOWN";
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────────────────────────────────────
static size_t cudaDtypeSize(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_8F_E4M3:  return 1;
        case CUDA_R_8F_E5M2:  return 1;
        case CUDA_R_16F:      return 2;
        case CUDA_R_16BF:     return 2;
        case CUDA_R_32F:      return 4;
        default:              return 4;
    }
}

static const char* cudaDtypeName(cudaDataType_t t) {
    switch (t) {
        case CUDA_R_8F_E4M3:  return "E4M3";
        case CUDA_R_8F_E5M2:  return "E5M2";
        case CUDA_R_16F:      return "FP16";
        case CUDA_R_16BF:     return "BF16";
        case CUDA_R_32F:      return "FP32";
        default:              return "???";
    }
}

static const char* computeTypeName(cublasComputeType_t t) {
    switch (t) {
        case CUBLAS_COMPUTE_16F:          return "COMPUTE_16F";
        case CUBLAS_COMPUTE_32F:          return "COMPUTE_32F";
        case CUBLAS_COMPUTE_32F_FAST_16F: return "COMPUTE_32F_FAST_16F";
        case CUBLAS_COMPUTE_32F_FAST_TF32:return "COMPUTE_32F_FAST_TF32";
        default:                          return "COMPUTE_???";
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Core test function
// Returns measured TFLOPS if successful, -1 if not supported, -2 on other error
// ──────────────────────────────────────────────────────────────────────────────
struct TestResult {
    cublasStatus_t status_desc;      // from matmulDescCreate
    cublasStatus_t status_layout;    // from matrixLayoutCreate (A)
    cublasStatus_t status_pref;      // from preferenceCreate
    cublasStatus_t status_heuristic; // from algoGetHeuristic
    int            num_algos;        // how many algos returned
    cublasStatus_t status_matmul;    // from actual matmul
    double         tflops;           // achieved (0 if not run)
};

static TestResult runGemmTest(
    cublasLtHandle_t lt,
    int M, int N, int K,
    cudaDataType_t typeA,
    cudaDataType_t typeB,
    cudaDataType_t typeC,
    cublasComputeType_t computeType,
    cudaDataType_t scaleType,
    void* dA, void* dB, void* dC, void* dWorkspace, size_t workspaceBytes
) {
    TestResult res = {};
    res.tflops = 0.0;

    // ---- Matmul descriptor ----
    cublasLtMatmulDesc_t opDesc = nullptr;
    res.status_desc = cublasLtMatmulDescCreate(&opDesc, computeType, scaleType);
    if (res.status_desc != CUBLAS_STATUS_SUCCESS) {
        return res;
    }

    // Set transpose ops (NN layout)
    cublasOperation_t transa = CUBLAS_OP_N, transb = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSA, &transa, sizeof(transa));
    cublasLtMatmulDescSetAttribute(opDesc, CUBLASLT_MATMUL_DESC_TRANSB, &transb, sizeof(transb));

    // ---- Matrix layouts ----
    cublasLtMatrixLayout_t layoutA = nullptr, layoutB = nullptr, layoutC = nullptr;

    res.status_layout = cublasLtMatrixLayoutCreate(&layoutA, typeA, M, K, M);
    if (res.status_layout != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatmulDescDestroy(opDesc);
        return res;
    }
    cublasLtMatrixLayoutCreate(&layoutB, typeB, K, N, K);
    cublasLtMatrixLayoutCreate(&layoutC, typeC, M, N, M);

    // ---- Preference ----
    cublasLtMatmulPreference_t pref = nullptr;
    res.status_pref = cublasLtMatmulPreferenceCreate(&pref);
    if (res.status_pref != CUBLAS_STATUS_SUCCESS) {
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(opDesc);
        return res;
    }
    cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
                                         &workspaceBytes, sizeof(workspaceBytes));

    // ---- Heuristic algo search ----
    const int maxAlgos = 8;
    cublasLtMatmulHeuristicResult_t heuristicResults[8] = {};
    int returnedAlgos = 0;

    res.status_heuristic = cublasLtMatmulAlgoGetHeuristic(
        lt, opDesc, layoutA, layoutB, layoutC, layoutC,
        pref, maxAlgos, heuristicResults, &returnedAlgos);
    res.num_algos = returnedAlgos;

    if (res.status_heuristic != CUBLAS_STATUS_SUCCESS || returnedAlgos == 0) {
        cublasLtMatmulPreferenceDestroy(pref);
        cublasLtMatrixLayoutDestroy(layoutA);
        cublasLtMatrixLayoutDestroy(layoutB);
        cublasLtMatrixLayoutDestroy(layoutC);
        cublasLtMatmulDescDestroy(opDesc);
        // Treat zero algos as NOT_SUPPORTED
        if (returnedAlgos == 0 && res.status_heuristic == CUBLAS_STATUS_SUCCESS)
            res.status_matmul = CUBLAS_STATUS_NOT_SUPPORTED;
        return res;
    }

    // ---- Warmup + timed run ----
    float alpha = 1.0f, beta = 0.0f;
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Warmup
    res.status_matmul = cublasLtMatmul(
        lt, opDesc,
        &alpha, dA, layoutA, dB, layoutB,
        &beta,  dC, layoutC, dC, layoutC,
        &heuristicResults[0].algo,
        dWorkspace, workspaceBytes, stream);

    if (res.status_matmul == CUBLAS_STATUS_SUCCESS) {
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // Timed runs
        const int NRUNS = 50;
        cudaEvent_t t0, t1;
        CUDA_CHECK(cudaEventCreate(&t0));
        CUDA_CHECK(cudaEventCreate(&t1));
        CUDA_CHECK(cudaEventRecord(t0, stream));
        for (int i = 0; i < NRUNS; i++) {
            cublasLtMatmul(
                lt, opDesc,
                &alpha, dA, layoutA, dB, layoutB,
                &beta,  dC, layoutC, dC, layoutC,
                &heuristicResults[0].algo,
                dWorkspace, workspaceBytes, stream);
        }
        CUDA_CHECK(cudaEventRecord(t1, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, t0, t1));
        double us = (ms / NRUNS) * 1000.0;
        double flops = 2.0 * (double)M * (double)N * (double)K;
        res.tflops = flops / (us * 1e6);   // TFLOPS
        CUDA_CHECK(cudaEventDestroy(t0));
        CUDA_CHECK(cudaEventDestroy(t1));
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(layoutA);
    cublasLtMatrixLayoutDestroy(layoutB);
    cublasLtMatrixLayoutDestroy(layoutC);
    cublasLtMatmulDescDestroy(opDesc);
    return res;
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────
int main() {
    printf("=== cuBLAS FP8 GEMM Check on B300 sm_103a (CUDA 13.2) ===\n\n");

    // Print CUDA / cuBLAS version
    int cudaVer = 0;
    cudaRuntimeGetVersion(&cudaVer);
    printf("CUDA runtime version: %d.%d\n", cudaVer/1000, (cudaVer%1000)/10);
    int cublasVer = 0;
    cublasGetVersion(nullptr, &cublasVer);
    printf("cuBLAS version: %d.%d.%d\n",
           cublasVer/100000, (cublasVer%100000)/1000, cublasVer%1000);

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s  (SM %d.%d)\n\n", prop.name, prop.major, prop.minor);

    // ---- Create cublasLt handle ----
    cublasLtHandle_t lt;
    cublasStatus_t st = cublasLtCreate(&lt);
    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasLtCreate failed: %s\n", cublasStatusStr(st));
        return 1;
    }

    // ---- Allocate device memory (max size = 8192² × 1 byte for FP8) ----
    const int MAXDIM = 8192;
    const size_t MAXBYTES = (size_t)MAXDIM * MAXDIM * 4;  // 4 bytes per element worst case
    void *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, MAXBYTES));
    CUDA_CHECK(cudaMalloc(&dB, MAXBYTES));
    CUDA_CHECK(cudaMalloc(&dC, MAXBYTES));
    CUDA_CHECK(cudaMemset(dA, 0, MAXBYTES));
    CUDA_CHECK(cudaMemset(dB, 0, MAXBYTES));
    CUDA_CHECK(cudaMemset(dC, 0, MAXBYTES));

    // ---- Workspace ----
    const size_t WS_BYTES = 32ULL * 1024 * 1024;  // 32 MB
    void *dWorkspace;
    CUDA_CHECK(cudaMalloc(&dWorkspace, WS_BYTES));
    CUDA_CHECK(cudaMemset(dWorkspace, 0, WS_BYTES));

    // ──────────────────────────────────────────────────────────────
    // 1. CONTROL: FP16 and BF16 GEMMs — must succeed
    // ──────────────────────────────────────────────────────────────
    printf("=== CONTROL: FP16 / BF16 GEMMs (must succeed) ===\n");
    printf("%-8s  %-8s  %-6s  %-4s  %-4s  %-13s  %-13s  %-13s  %s\n",
           "TypeA", "TypeC", "Compute", "M", "K",
           "DescStatus", "HeurStatus", "MatmulStatus", "TFLOPS");

    struct ControlCase {
        cudaDataType_t typeAB;
        cudaDataType_t typeC;
        cublasComputeType_t computeType;
        cudaDataType_t scaleType;
        const char* label;
    };
    ControlCase controls[] = {
        { CUDA_R_16F,  CUDA_R_16F,  CUBLAS_COMPUTE_16F,           CUDA_R_32F, "FP16→FP16" },
        { CUDA_R_16F,  CUDA_R_32F,  CUBLAS_COMPUTE_32F_FAST_16F,  CUDA_R_32F, "FP16→FP32" },
        { CUDA_R_16BF, CUDA_R_16BF, CUBLAS_COMPUTE_32F_FAST_16F,  CUDA_R_32F, "BF16→BF16" },
        { CUDA_R_16BF, CUDA_R_32F,  CUBLAS_COMPUTE_32F_FAST_16F,  CUDA_R_32F, "BF16→FP32" },
        { CUDA_R_32F,  CUDA_R_32F,  CUBLAS_COMPUTE_32F_FAST_TF32, CUDA_R_32F, "TF32→FP32" },
    };

    for (auto& c : controls) {
        int M = 4096, K = 4096, N = 4096;
        TestResult r = runGemmTest(lt, M, N, K,
            c.typeAB, c.typeAB, c.typeC,
            c.computeType, c.scaleType,
            dA, dB, dC, dWorkspace, WS_BYTES);
        printf("%-20s M=%d  desc=%-16s heur=%-16s matmul=%-16s algos=%d  %.1f TFLOPS\n",
               c.label, M,
               cublasStatusStr(r.status_desc),
               cublasStatusStr(r.status_heuristic),
               cublasStatusStr(r.status_matmul),
               r.num_algos,
               r.tflops);
    }

    // ──────────────────────────────────────────────────────────────
    // 2. FP8 E4M3/E5M2 × various outputs
    // ──────────────────────────────────────────────────────────────
    printf("\n=== FP8 TESTS ===\n");
    printf("Testing all FP8 input/output/compute combinations.\n");
    printf("If any show SUCCESS that contradicts the 'Not Available' catalog section.\n\n");

    struct FP8Case {
        cudaDataType_t typeA;
        cudaDataType_t typeB;
        cudaDataType_t typeC;
        cublasComputeType_t computeType;
        cudaDataType_t scaleType;
    };

    FP8Case fp8cases[] = {
        // E4M3 × E4M3 → various outputs, COMPUTE_32F
        { CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F },
        { CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_16F,  CUBLAS_COMPUTE_32F, CUDA_R_32F },
        { CUDA_R_8F_E4M3, CUDA_R_8F_E4M3, CUDA_R_32F,  CUBLAS_COMPUTE_32F, CUDA_R_32F },
        // E4M3 × E5M2 → BF16/F32
        { CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F },
        { CUDA_R_8F_E4M3, CUDA_R_8F_E5M2, CUDA_R_32F,  CUBLAS_COMPUTE_32F, CUDA_R_32F },
        // E5M2 × E5M2 → BF16/F32
        { CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F },
        { CUDA_R_8F_E5M2, CUDA_R_8F_E5M2, CUDA_R_32F,  CUBLAS_COMPUTE_32F, CUDA_R_32F },
        // E5M2 × E4M3 → BF16
        { CUDA_R_8F_E5M2, CUDA_R_8F_E4M3, CUDA_R_16BF, CUBLAS_COMPUTE_32F, CUDA_R_32F },
    };

    int sizes[] = { 1024, 2048, 4096, 8192 };

    printf("%-8s  %-8s  %-8s  %-22s  %-6s  %-16s  %-16s  %-16s  %s\n",
           "TypeA", "TypeB", "TypeC", "Compute", "M=N=K",
           "DescStatus", "HeurStatus", "MatmulStatus", "TFLOPS/algos");

    bool anyFP8Success = false;

    for (auto& c : fp8cases) {
        for (int sz : sizes) {
            TestResult r = runGemmTest(lt, sz, sz, sz,
                c.typeA, c.typeB, c.typeC,
                c.computeType, c.scaleType,
                dA, dB, dC, dWorkspace, WS_BYTES);

            bool success = (r.status_matmul == CUBLAS_STATUS_SUCCESS);
            bool notSupp = (r.status_heuristic != CUBLAS_STATUS_SUCCESS && r.num_algos == 0)
                         || (r.status_desc != CUBLAS_STATUS_SUCCESS)
                         || (r.status_matmul == CUBLAS_STATUS_NOT_SUPPORTED);

            if (success) anyFP8Success = true;

            printf("%-8s  %-8s  %-8s  %-22s  %-6d  %-16s  %-16s  %-16s",
                   cudaDtypeName(c.typeA), cudaDtypeName(c.typeB), cudaDtypeName(c.typeC),
                   computeTypeName(c.computeType), sz,
                   cublasStatusStr(r.status_desc),
                   (r.status_desc != CUBLAS_STATUS_SUCCESS) ? "N/A" : cublasStatusStr(r.status_heuristic),
                   (r.status_desc != CUBLAS_STATUS_SUCCESS || r.status_heuristic != CUBLAS_STATUS_SUCCESS) ? "N/A" : cublasStatusStr(r.status_matmul));

            if (success) {
                printf("  %.1f TFLOPS  *** SUCCESS ***\n", r.tflops);
            } else {
                printf("  algos=%d\n", r.num_algos);
            }
        }
        printf("\n");
    }

    // ──────────────────────────────────────────────────────────────
    // 3. Summary
    // ──────────────────────────────────────────────────────────────
    printf("=== SUMMARY ===\n");
    if (anyFP8Success) {
        printf("RESULT: cuBLAS FP8 GEMM IS SUPPORTED on this B300 (sm_%d%d) with CUDA %d.%d\n",
               prop.major, prop.minor, cudaVer/1000, (cudaVer%1000)/10);
        printf("        => The 'FP8 cuBLAS: Not Available' catalog section is WRONG.\n");
    } else {
        printf("RESULT: cuBLAS FP8 GEMM is NOT SUPPORTED via standard cublasLtMatmul API.\n");
        printf("        => Catalog section 'FP8 cuBLAS: Not Available' is CORRECT.\n");
        printf("        => Earlier catalog FP8 TFLOPS numbers came from direct tcgen05.mma kernels.\n");
    }

    // ──────────────────────────────────────────────────────────────
    // 4. Cross-check: FP8-sized BF16 for baseline
    // ──────────────────────────────────────────────────────────────
    printf("\n=== CROSS-CHECK: BF16 at same shapes (to confirm measurement methodology) ===\n");
    printf("%-6s  %s\n", "M=N=K", "BF16→BF16 TFLOPS");
    for (int sz : sizes) {
        TestResult r = runGemmTest(lt, sz, sz, sz,
            CUDA_R_16BF, CUDA_R_16BF, CUDA_R_16BF,
            CUBLAS_COMPUTE_32F_FAST_16F, CUDA_R_32F,
            dA, dB, dC, dWorkspace, WS_BYTES);
        printf("%-6d  %.1f TFLOPS  (%s)\n", sz,
               r.tflops, cublasStatusStr(r.status_matmul));
    }

    // Cleanup
    cublasLtDestroy(lt);
    cudaFree(dA); cudaFree(dB); cudaFree(dC); cudaFree(dWorkspace);

    printf("\nDone.\n");
    return 0;
}
