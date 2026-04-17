// atomic_matrix_runner.cu
// Standalone program: measures all (scope x ordering) atomic costs on B300.
// Compiles with: nvcc -arch=sm_103a -O3 -o atomic_matrix_runner atomic_matrix_runner.cu
//
// Each test case is a separate __global__ function so the compiler generates
// correct code for each scope+ordering combination.
//
// Methodology:
//   - 1 block, 32 threads (single warp, no inter-warp contention)
//   - ITERS atomics per thread, chained through return value (latency measure)
//   - Each thread has its own address slot (no cross-thread contention)
//   - clock64 timing, thread 0 stores result
//   - Clock locked to 2032 MHz externally

#include <cstdio>
#include <cstdlib>
#include <climits>
#include <cuda_runtime.h>

#define ITERS 4096
#define NRUNS 5

#define CHECK(x) do { \
    cudaError_t e = (x); \
    if (e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(1); \
    } \
} while(0)

// ===== KERNEL DEFINITIONS =====
// Each kernel uses a chain: v = r + 1 to force true latency measurement.
// Thread t operates on slot t in shared/global memory.

// Helper macro to define a shared-mem atom kernel
#define DEF_SMEM_KERNEL(NAME, ASM) \
__global__ __launch_bounds__(32, 1) \
void NAME(unsigned* A, unsigned long long* result) { \
    extern __shared__ unsigned smem[]; \
    if (threadIdx.x < 64) smem[threadIdx.x] = threadIdx.x; \
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

// Helper macro to define a global-mem atom kernel
#define DEF_GMEM_KERNEL(NAME, ASM) \
__global__ __launch_bounds__(32, 1) \
void NAME(unsigned* A, unsigned long long* result) { \
    unsigned long long gaddr = (unsigned long long)A + (unsigned long long)threadIdx.x * 4ULL; \
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
    if (v == 0xDEADBEEFu) A[32 + threadIdx.x] = v; \
}

// ===== SHARED MEMORY KERNELS =====

// Baseline: no scope/ordering qualifier
DEF_SMEM_KERNEL(smem_bare,
    "atom.shared.add.u32 %0, [%1], %2;")

// relaxed
DEF_SMEM_KERNEL(smem_relaxed_cta,
    "atom.relaxed.cta.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_relaxed_cluster,
    "atom.relaxed.cluster.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_relaxed_gpu,
    "atom.relaxed.gpu.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_relaxed_sys,
    "atom.relaxed.sys.shared.add.u32 %0, [%1], %2;")

// acquire
DEF_SMEM_KERNEL(smem_acquire_cta,
    "atom.acquire.cta.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acquire_cluster,
    "atom.acquire.cluster.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acquire_gpu,
    "atom.acquire.gpu.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acquire_sys,
    "atom.acquire.sys.shared.add.u32 %0, [%1], %2;")

// release
DEF_SMEM_KERNEL(smem_release_cta,
    "atom.release.cta.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_release_cluster,
    "atom.release.cluster.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_release_gpu,
    "atom.release.gpu.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_release_sys,
    "atom.release.sys.shared.add.u32 %0, [%1], %2;")

// acq_rel
DEF_SMEM_KERNEL(smem_acqrel_cta,
    "atom.acq_rel.cta.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acqrel_cluster,
    "atom.acq_rel.cluster.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acqrel_gpu,
    "atom.acq_rel.gpu.shared.add.u32 %0, [%1], %2;")
DEF_SMEM_KERNEL(smem_acqrel_sys,
    "atom.acq_rel.sys.shared.add.u32 %0, [%1], %2;")

// ===== GLOBAL MEMORY KERNELS =====

// Baseline: no scope/ordering qualifier
DEF_GMEM_KERNEL(gmem_bare,
    "atom.global.add.u32 %0, [%1], %2;")

// relaxed
DEF_GMEM_KERNEL(gmem_relaxed_cta,
    "atom.relaxed.cta.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_relaxed_cluster,
    "atom.relaxed.cluster.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_relaxed_gpu,
    "atom.relaxed.gpu.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_relaxed_sys,
    "atom.relaxed.sys.global.add.u32 %0, [%1], %2;")

// acquire
DEF_GMEM_KERNEL(gmem_acquire_cta,
    "atom.acquire.cta.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acquire_cluster,
    "atom.acquire.cluster.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acquire_gpu,
    "atom.acquire.gpu.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acquire_sys,
    "atom.acquire.sys.global.add.u32 %0, [%1], %2;")

// release
DEF_GMEM_KERNEL(gmem_release_cta,
    "atom.release.cta.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_release_cluster,
    "atom.release.cluster.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_release_gpu,
    "atom.release.gpu.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_release_sys,
    "atom.release.sys.global.add.u32 %0, [%1], %2;")

// acq_rel
DEF_GMEM_KERNEL(gmem_acqrel_cta,
    "atom.acq_rel.cta.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acqrel_cluster,
    "atom.acq_rel.cluster.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acqrel_gpu,
    "atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;")
DEF_GMEM_KERNEL(gmem_acqrel_sys,
    "atom.acq_rel.sys.global.add.u32 %0, [%1], %2;")

// NOTE: seq_cst is not supported by ptxas sm_103a for atom instructions.
// The PTX ISA docs list it but ptxas rejects it with "Unknown modifier '.seq_cst'"

// ===== HOST CODE =====

typedef void (*kernel_fn)(unsigned*, unsigned long long*);

struct TestCase {
    const char* mem;
    const char* ordering;
    const char* scope;
    kernel_fn fn;
};

#define TC(mem, ord, sc, fn) {mem, ord, sc, fn}

static TestCase tests[] = {
    // Shared memory
    TC("smem", "bare(=relaxed.cta)", "cta",     smem_bare),
    TC("smem", "relaxed",            "cta",     smem_relaxed_cta),
    TC("smem", "relaxed",            "cluster", smem_relaxed_cluster),
    TC("smem", "relaxed",            "gpu",     smem_relaxed_gpu),
    TC("smem", "relaxed",            "sys",     smem_relaxed_sys),
    TC("smem", "acquire",            "cta",     smem_acquire_cta),
    TC("smem", "acquire",            "cluster", smem_acquire_cluster),
    TC("smem", "acquire",            "gpu",     smem_acquire_gpu),
    TC("smem", "acquire",            "sys",     smem_acquire_sys),
    TC("smem", "release",            "cta",     smem_release_cta),
    TC("smem", "release",            "cluster", smem_release_cluster),
    TC("smem", "release",            "gpu",     smem_release_gpu),
    TC("smem", "release",            "sys",     smem_release_sys),
    TC("smem", "acq_rel",            "cta",     smem_acqrel_cta),
    TC("smem", "acq_rel",            "cluster", smem_acqrel_cluster),
    TC("smem", "acq_rel",            "gpu",     smem_acqrel_gpu),
    TC("smem", "acq_rel",            "sys",     smem_acqrel_sys),
    // Global memory
    TC("gmem", "bare(=relaxed.gpu)", "gpu",     gmem_bare),
    TC("gmem", "relaxed",            "cta",     gmem_relaxed_cta),
    TC("gmem", "relaxed",            "cluster", gmem_relaxed_cluster),
    TC("gmem", "relaxed",            "gpu",     gmem_relaxed_gpu),
    TC("gmem", "relaxed",            "sys",     gmem_relaxed_sys),
    TC("gmem", "acquire",            "cta",     gmem_acquire_cta),
    TC("gmem", "acquire",            "cluster", gmem_acquire_cluster),
    TC("gmem", "acquire",            "gpu",     gmem_acquire_gpu),
    TC("gmem", "acquire",            "sys",     gmem_acquire_sys),
    TC("gmem", "release",            "cta",     gmem_release_cta),
    TC("gmem", "release",            "cluster", gmem_release_cluster),
    TC("gmem", "release",            "gpu",     gmem_release_gpu),
    TC("gmem", "release",            "sys",     gmem_release_sys),
    TC("gmem", "acq_rel",            "cta",     gmem_acqrel_cta),
    TC("gmem", "acq_rel",            "cluster", gmem_acqrel_cluster),
    TC("gmem", "acq_rel",            "gpu",     gmem_acqrel_gpu),
    TC("gmem", "acq_rel",            "sys",     gmem_acqrel_sys),
};

static const int NTESTS = sizeof(tests) / sizeof(tests[0]);

int main() {
    // Allocate device memory
    unsigned* d_A = nullptr;
    unsigned long long* d_result = nullptr;
    CHECK(cudaMalloc(&d_A, 256 * sizeof(unsigned)));
    CHECK(cudaMalloc(&d_result, sizeof(unsigned long long)));
    CHECK(cudaMemset(d_A, 0, 256 * sizeof(unsigned)));

    // Warmup
    smem_bare<<<1, 32, 1024 * sizeof(unsigned)>>>(d_A, d_result);
    CHECK(cudaDeviceSynchronize());

    printf("%-6s  %-24s  %-8s  %8s  %8s  %8s\n",
           "mem", "ordering", "scope", "cy/iter", "ns@2032", "x_relaxed");
    printf("%-6s  %-24s  %-8s  %8s  %8s  %8s\n",
           "------", "------------------------", "--------", "--------", "-------", "---------");

    // Reference values for normalization (relaxed.cta for smem, relaxed.gpu for gmem)
    float smem_ref_cy = 0.0f, gmem_ref_cy = 0.0f;
    bool smem_ref_set = false, gmem_ref_set = false;

    // Store results for second pass normalization
    float cy_results[NTESTS];

    for (int t = 0; t < NTESTS; t++) {
        TestCase& tc = tests[t];
        unsigned long long h_result;
        unsigned long long min_cy = ULLONG_MAX;

        for (int run = 0; run < NRUNS; run++) {
            CHECK(cudaMemset(d_A, 0, 256 * sizeof(unsigned)));
            tc.fn<<<1, 32, 256 * sizeof(unsigned)>>>(d_A, d_result);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(&h_result, d_result, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            if (h_result < min_cy) min_cy = h_result;
        }

        float cy_per = (float)min_cy / (float)ITERS;
        float ns_per = cy_per / 2.032f;
        cy_results[t] = cy_per;

        // Set reference values
        if (!smem_ref_set && strcmp(tc.mem, "smem") == 0 &&
            strstr(tc.ordering, "relaxed") != nullptr && strcmp(tc.scope, "cta") == 0) {
            smem_ref_cy = cy_per;
            smem_ref_set = true;
        }
        if (!gmem_ref_set && strcmp(tc.mem, "gmem") == 0 &&
            strstr(tc.ordering, "relaxed") != nullptr && strcmp(tc.scope, "gpu") == 0) {
            gmem_ref_cy = cy_per;
            gmem_ref_set = true;
        }
    }

    // Also handle bare baselines
    // smem bare is index 0, gmem bare is index 17
    float smem_bare_cy = cy_results[0];
    float gmem_bare_cy = cy_results[17];

    // Print results with normalization
    for (int t = 0; t < NTESTS; t++) {
        TestCase& tc = tests[t];
        float cy_per = cy_results[t];
        float ns_per = cy_per / 2.032f;

        float ref = (strcmp(tc.mem, "smem") == 0) ? smem_ref_cy : gmem_ref_cy;
        float ratio = (ref > 0) ? cy_per / ref : 0.0f;

        printf("%-6s  %-24s  %-8s  %8.1f  %8.2f  %8.2f\n",
               tc.mem, tc.ordering, tc.scope, cy_per, ns_per, ratio);
    }

    printf("\nReference: smem relaxed.cta = %.1f cy, gmem relaxed.gpu = %.1f cy\n",
           smem_ref_cy, gmem_ref_cy);
    printf("Baseline:  smem bare(noqualifier)=%.1f cy, gmem bare(noqualifier)=%.1f cy\n",
           smem_bare_cy, gmem_bare_cy);
    printf("Clock: ~2032 MHz (%.2f GHz)\n", 2.032f);

    CHECK(cudaFree(d_A));
    CHECK(cudaFree(d_result));
    return 0;
}
