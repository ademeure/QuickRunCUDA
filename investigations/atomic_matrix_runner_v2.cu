// atomic_matrix_runner_v2.cu
// Comprehensive (scope x ordering) cost matrix for B300 atomics.
// v2: More runs for stability, separate warmup, better global-mem setup.
//
// KEY INSIGHT on methodology:
//   - smem: each thread has its own slot -> no contention, pure latency
//   - gmem: each thread has its own cacheline slot -> L2-hit latency
//           (first access will miss, subsequent hits after warmup)
//   - sys-scope atomics are inherently variable due to system-wide bus traffic

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>

#define ITERS 8192
#define NRUNS  10     // take min of 10 runs for stable measurements

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ===== KERNEL MACROS =====
// Each thread operates on its own address slot (threadIdx.x).
// Chain: v = r + 1 forces true latency measurement (each atom waits for prev result).
// Anti-DCE: final v stored conditionally.

#define DEF_SMEM_KERNEL(NAME, ASM) \
__global__ __launch_bounds__(32, 1) \
void NAME(unsigned* A, unsigned long long* result) { \
    extern __shared__ unsigned smem[]; \
    smem[threadIdx.x] = threadIdx.x; \
    __syncthreads(); \
    unsigned saddr = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x]); \
    unsigned v = threadIdx.x + 1; \
    unsigned long long t0, t1; \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)); \
    _Pragma("unroll 1") \
    for (int i = 0; i < ITERS; i++) { \
        unsigned r; \
        asm volatile(ASM : "=r"(r) : "r"(saddr), "r"(v)); \
        v = r + 1u; \
    } \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)); \
    if (threadIdx.x == 0) *result = t1 - t0; \
    if (v == 0xDEADBEEFu) A[threadIdx.x] = v; \
}

// Global: each thread gets its own 128-byte-aligned cacheline to avoid cross-thread atomics.
// Thread t -> A + t * 32 (each slot is 32 words = 128 bytes = 1 cache line).
// This ensures all accesses hit different cachelines -> pure L2-hit latency (after warmup).
#define DEF_GMEM_KERNEL(NAME, ASM) \
__global__ __launch_bounds__(32, 1) \
void NAME(unsigned* A, unsigned long long* result) { \
    unsigned long long gaddr = (unsigned long long)A + (unsigned long long)threadIdx.x * 128ULL; \
    unsigned v = threadIdx.x + 1; \
    unsigned long long t0, t1; \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0)); \
    _Pragma("unroll 1") \
    for (int i = 0; i < ITERS; i++) { \
        unsigned r; \
        asm volatile(ASM : "=r"(r) : "l"(gaddr), "r"(v)); \
        v = r + 1u; \
    } \
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)); \
    if (threadIdx.x == 0) *result = t1 - t0; \
    if (v == 0xDEADBEEFu) A[1024 + threadIdx.x] = v; \
}

// ===== SHARED MEMORY KERNELS =====
DEF_SMEM_KERNEL(smem_bare,    "atom.shared.add.u32 %0, [%1], %2;")

DEF_SMEM_KERNEL(smem_rlx_cta,     "atom.relaxed.cta.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_rlx_cluster, "atom.relaxed.cluster.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_rlx_gpu,     "atom.relaxed.gpu.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_rlx_sys,     "atom.relaxed.sys.shared.add.u32 %0, [%1], %2;")

DEF_SMEM_KERNEL(smem_acq_cta,     "atom.acquire.cta.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acq_cluster, "atom.acquire.cluster.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acq_gpu,     "atom.acquire.gpu.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acq_sys,     "atom.acquire.sys.shared.add.u32 %0, [%1], %2;")

DEF_SMEM_KERNEL(smem_rel_cta,     "atom.release.cta.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_rel_cluster, "atom.release.cluster.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_rel_gpu,     "atom.release.gpu.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_rel_sys,     "atom.release.sys.shared.add.u32 %0, [%1], %2;")

DEF_SMEM_KERNEL(smem_ar_cta,      "atom.acq_rel.cta.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_ar_cluster,  "atom.acq_rel.cluster.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_ar_gpu,      "atom.acq_rel.gpu.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_ar_sys,      "atom.acq_rel.sys.shared.add.u32 %0, [%1], %2;")

// ===== GLOBAL MEMORY KERNELS =====
DEF_GMEM_KERNEL(gmem_bare,    "atom.global.add.u32 %0, [%1], %2;")

DEF_GMEM_KERNEL(gmem_rlx_cta,     "atom.relaxed.cta.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_rlx_cluster, "atom.relaxed.cluster.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_rlx_gpu,     "atom.relaxed.gpu.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_rlx_sys,     "atom.relaxed.sys.global.add.u32 %0, [%1], %2;")

DEF_GMEM_KERNEL(gmem_acq_cta,     "atom.acquire.cta.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acq_cluster, "atom.acquire.cluster.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acq_gpu,     "atom.acquire.gpu.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acq_sys,     "atom.acquire.sys.global.add.u32 %0, [%1], %2;")

DEF_GMEM_KERNEL(gmem_rel_cta,     "atom.release.cta.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_rel_cluster, "atom.release.cluster.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_rel_gpu,     "atom.release.gpu.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_rel_sys,     "atom.release.sys.global.add.u32 %0, [%1], %2;")

DEF_GMEM_KERNEL(gmem_ar_cta,      "atom.acq_rel.cta.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_ar_cluster,  "atom.acq_rel.cluster.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_ar_gpu,      "atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_ar_sys,      "atom.acq_rel.sys.global.add.u32 %0, [%1], %2;")

// ===== HOST =====
typedef void (*kernel_fn)(unsigned*, unsigned long long*);

struct TestCase {
    const char* mem;
    const char* ordering;
    const char* scope;
    kernel_fn fn;
};

static TestCase tests[] = {
    // Shared
    {"smem", "bare(=rlx.cta)", "cta",     smem_bare},
    {"smem", "relaxed",        "cta",     smem_rlx_cta},
    {"smem", "relaxed",        "cluster", smem_rlx_cluster},
    {"smem", "relaxed",        "gpu",     smem_rlx_gpu},
    {"smem", "relaxed",        "sys",     smem_rlx_sys},
    {"smem", "acquire",        "cta",     smem_acq_cta},
    {"smem", "acquire",        "cluster", smem_acq_cluster},
    {"smem", "acquire",        "gpu",     smem_acq_gpu},
    {"smem", "acquire",        "sys",     smem_acq_sys},
    {"smem", "release",        "cta",     smem_rel_cta},
    {"smem", "release",        "cluster", smem_rel_cluster},
    {"smem", "release",        "gpu",     smem_rel_gpu},
    {"smem", "release",        "sys",     smem_rel_sys},
    {"smem", "acq_rel",        "cta",     smem_ar_cta},
    {"smem", "acq_rel",        "cluster", smem_ar_cluster},
    {"smem", "acq_rel",        "gpu",     smem_ar_gpu},
    {"smem", "acq_rel",        "sys",     smem_ar_sys},
    // Global
    {"gmem", "bare(=rlx.gpu)", "gpu",     gmem_bare},
    {"gmem", "relaxed",        "cta",     gmem_rlx_cta},
    {"gmem", "relaxed",        "cluster", gmem_rlx_cluster},
    {"gmem", "relaxed",        "gpu",     gmem_rlx_gpu},
    {"gmem", "relaxed",        "sys",     gmem_rlx_sys},
    {"gmem", "acquire",        "cta",     gmem_acq_cta},
    {"gmem", "acquire",        "cluster", gmem_acq_cluster},
    {"gmem", "acquire",        "gpu",     gmem_acq_gpu},
    {"gmem", "acquire",        "sys",     gmem_acq_sys},
    {"gmem", "release",        "cta",     gmem_rel_cta},
    {"gmem", "release",        "cluster", gmem_rel_cluster},
    {"gmem", "release",        "gpu",     gmem_rel_gpu},
    {"gmem", "release",        "sys",     gmem_rel_sys},
    {"gmem", "acq_rel",        "cta",     gmem_ar_cta},
    {"gmem", "acq_rel",        "cluster", gmem_ar_cluster},
    {"gmem", "acq_rel",        "gpu",     gmem_ar_gpu},
    {"gmem", "acq_rel",        "sys",     gmem_ar_sys},
};
static const int NTESTS = sizeof(tests)/sizeof(tests[0]);

int main() {
    // 32 threads * 128 bytes stride * 2 (extra) = 8192 bytes for global slots
    unsigned* d_A;
    unsigned long long* d_result;
    CHECK(cudaMalloc(&d_A, 8192 * sizeof(unsigned)));
    CHECK(cudaMalloc(&d_result, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_A, 0, 8192 * sizeof(unsigned)));

    // Warmup: run each kernel once to prime caches and ramp clock
    for (int t = 0; t < NTESTS; t++) {
        tests[t].fn<<<1, 32, 256*sizeof(unsigned)>>>(d_A, d_result);
    }
    CHECK(cudaDeviceSynchronize());

    float cy_results[NTESTS];
    unsigned long long all_results[NTESTS][NRUNS];

    // Timed runs
    for (int run = 0; run < NRUNS; run++) {
        for (int t = 0; t < NTESTS; t++) {
            CHECK(cudaMemset(d_A, 0, 8192 * sizeof(unsigned)));
            tests[t].fn<<<1, 32, 256*sizeof(unsigned)>>>(d_A, d_result);
            CHECK(cudaDeviceSynchronize());
            unsigned long long h_result;
            CHECK(cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            all_results[t][run] = h_result;
        }
    }

    // Compute min over runs
    for (int t = 0; t < NTESTS; t++) {
        unsigned long long min_cy = ULLONG_MAX;
        for (int run = 0; run < NRUNS; run++) {
            if (all_results[t][run] < min_cy) min_cy = all_results[t][run];
        }
        cy_results[t] = (float)min_cy / (float)ITERS;
    }

    // Reference values
    float smem_ref = 0, gmem_ref = 0;
    for (int t = 0; t < NTESTS; t++) {
        if (strcmp(tests[t].mem, "smem") == 0 && strcmp(tests[t].ordering, "relaxed") == 0 && strcmp(tests[t].scope, "cta") == 0)
            smem_ref = cy_results[t];
        if (strcmp(tests[t].mem, "gmem") == 0 && strcmp(tests[t].ordering, "relaxed") == 0 && strcmp(tests[t].scope, "gpu") == 0)
            gmem_ref = cy_results[t];
    }

    // Print header
    printf("\n=== B300 Atomic Scope x Ordering Cost Matrix ===\n");
    printf("Clock: 2032 MHz (verified by ns/cy ratio)\n");
    printf("Method: single warp (32 threads), each thread its own address, ITERS=%d, min of %d runs\n\n",
           ITERS, NRUNS);

    printf("%-6s  %-22s  %-8s  %8s  %8s  %10s\n",
           "Mem", "Ordering", "Scope", "cy/iter", "ns@2032", "x_base");
    printf("%-6s  %-22s  %-8s  %8s  %8s  %10s\n",
           "------", "----------------------", "--------", "--------", "--------", "----------");

    for (int t = 0; t < NTESTS; t++) {
        float ref = (strcmp(tests[t].mem, "smem") == 0) ? smem_ref : gmem_ref;
        float ratio = (ref > 0) ? cy_results[t] / ref : 0.0f;
        float ns = cy_results[t] / 2.032f;
        printf("%-6s  %-22s  %-8s  %8.1f  %8.2f  %10.2f\n",
               tests[t].mem, tests[t].ordering, tests[t].scope,
               cy_results[t], ns, ratio);
    }

    printf("\nReference: smem relaxed.cta = %.1f cy,  gmem relaxed.gpu = %.1f cy\n",
           smem_ref, gmem_ref);

    // Also print raw run data for sys cases (high variance expected)
    printf("\nRaw run data for high-variance cases (sys scope, cy/iter):\n");
    for (int t = 0; t < NTESTS; t++) {
        if (strcmp(tests[t].scope, "sys") == 0) {
            printf("  %s.%s.%s: ", tests[t].mem, tests[t].ordering, tests[t].scope);
            for (int run = 0; run < NRUNS; run++) {
                printf("%.0f ", (float)all_results[t][run] / ITERS);
            }
            printf("\n");
        }
    }

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_result));
    return 0;
}
