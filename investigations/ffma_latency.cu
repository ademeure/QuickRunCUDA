// ffma_latency.cu — Definitive FFMA latency measurement on B300 sm_103a
//
// PURPOSE: Resolve the "4 cy vs 23 cy" FFMA latency question.
//
// METHODOLOGY:
//   - Single warp (32 threads), only thread 0 does the measurement timing
//   - All variants use asm volatile PTX to guarantee FFMA emission
//   - b and c operands loaded from __shared__ at runtime (opaque to compiler)
//   - Pragma unroll 1 on outer loop, fully unroll inner for clean SASS chains
//   - clock64 sampled before/after the chain
//   - Anti-DCE: result written via impossible-predicate guarded store
//
// VARIANTS TESTED:
//   1. "self-op" single chain: fma a,a,a,0    (likely inflated ~2x)
//   2. "self+const" single chain: fma a,a,b,c  (a=src+dst, b/c from shmem)
//   3. "diff sources" single chain: fma a,b,c,a (a=dst+addend, b=mul from shmem)
//   4-6. 2-chain, 4-chain, 8-chain ILP of variant 2 (throughput scaling)
//
// COMPILE:
//   nvcc -arch=sm_103a -O3 -o ffma_latency ffma_latency.cu
//
// RUN:
//   nvidia-smi -lgc 2032
//   ./ffma_latency
//   nvidia-smi -rgc

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { \
    cudaError_t _e = (x); \
    if (_e != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(_e)); \
        exit(1); \
    } \
} while(0)

// Number of FMAs per inner unrolled block (enough for clean measurement)
// We measure in 2 chunks: INNER * OUTER total ops
#define INNER 1024
#define OUTER 20

// ---------------------------------------------------------------------------
// Variant 1: Self-op chain  fma a,a,a,0
//   Each op: a = a*a + 0.0  (all three float sources are the same register)
//   Known to inflate latency 2x due to register read-port contention on Volta+
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_selfop(
    float* __restrict__ out_result,
    unsigned long long* __restrict__ out_timing,
    float init_val)
{
    // Only thread 0 does timing; all threads compute to keep warp active
    unsigned tid = threadIdx.x;

    float a = init_val + (float)tid * 0.001f;  // distinct per thread

    unsigned long long t0, t1;

    // Warm-up pass (not timed) — ensures no first-issue penalty
    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            // fma.rn.f32 a, a, a, 0  → a = a*a + 0
            asm volatile("fma.rn.f32 %0, %0, %0, 0f00000000;" : "+f"(a));
        }
    }

    // Timed pass
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            asm volatile("fma.rn.f32 %0, %0, %0, 0f00000000;" : "+f"(a));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    // Anti-DCE: impossible store (tid >= blockDim.x is always false)
    if (__float_as_int(a) == 0x7FFFFFFF) {
        out_result[tid] = a;
    }

    if (tid == 0) {
        out_timing[0] = t1 - t0;
    }
}

// ---------------------------------------------------------------------------
// Variant 2: Self+const chain  fma a, a, b, c
//   a = a*b + c where b,c come from __shared__ (runtime opaque)
//   'a' is both source and destination — still a RAW dep chain
//   b, c are NOT a — avoids the self-op port-pressure issue on the mul/add slots
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_self_const(
    float* __restrict__ out_result,
    unsigned long long* __restrict__ out_timing,
    float init_val)
{
    __shared__ float shmem[4];
    unsigned tid = threadIdx.x;

    // Runtime-initialized shmem so compiler can't fold b,c as constants
    if (tid == 0) {
        shmem[0] = 1.0001f;   // b: multiplier (close to 1 to avoid overflow)
        shmem[1] = 0.00001f;  // c: addend (small to avoid overflow)
    }
    __syncwarp();

    float b = shmem[0];
    float c = shmem[1];
    float a = init_val + (float)tid * 0.001f;

    unsigned long long t0, t1;

    // Warm-up
    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            // fma.rn.f32 a, a, b, c  → a = a*b + c
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a) : "f"(b), "f"(c));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    if (__float_as_int(a) == 0x7FFFFFFF) out_result[tid] = a;
    if (tid == 0) out_timing[1] = t1 - t0;
}

// ---------------------------------------------------------------------------
// Variant 3: Diff-sources chain  fma a, b, c, a
//   a = b*c + a where b,c from shmem (read-only), a is the accumulator
//   This is the canonical "FMA with running accumulator" form.
//   b and c are NOT involved in the RAW chain — only 'a' is.
// ---------------------------------------------------------------------------
__global__ __launch_bounds__(32, 1)
void kernel_diffsrc(
    float* __restrict__ out_result,
    unsigned long long* __restrict__ out_timing,
    float init_val)
{
    __shared__ float shmem[4];
    unsigned tid = threadIdx.x;

    if (tid == 0) {
        shmem[0] = 1.0001f;  // b
        shmem[1] = 1.0001f;  // c — product b*c ~1.0002
    }
    __syncwarp();

    float b = shmem[0];
    float c = shmem[1];
    float a = init_val + (float)tid * 0.001f;

    unsigned long long t0, t1;

    // Warm-up
    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            // fma.rn.f32 a, b, c, a  → a = b*c + a
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(a) : "f"(b), "f"(c));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) {
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(a) : "f"(b), "f"(c));
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    if (__float_as_int(a) == 0x7FFFFFFF) out_result[tid] = a;
    if (tid == 0) out_timing[2] = t1 - t0;
}

// ---------------------------------------------------------------------------
// ILP variants: N independent chains, all using the self+const form
//   fma a_k, a_k, b, c  for k in 0..N-1
//   Measures throughput (latency-hidden) not raw latency
// ---------------------------------------------------------------------------

// Helper macro: emit N chains of self+const FMA
#define EMIT_CHAINS_1(b,c)  \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a0) : "f"(b), "f"(c));

#define EMIT_CHAINS_2(b,c)  \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a0) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a1) : "f"(b), "f"(c));

#define EMIT_CHAINS_4(b,c)  \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a0) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a1) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a2) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a3) : "f"(b), "f"(c));

#define EMIT_CHAINS_8(b,c)  \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a0) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a1) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a2) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a3) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a4) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a5) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a6) : "f"(b), "f"(c)); \
    asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(a7) : "f"(b), "f"(c));

// 2-chain ILP
__global__ __launch_bounds__(32, 1)
void kernel_ilp2(
    float* __restrict__ out_result,
    unsigned long long* __restrict__ out_timing,
    float init_val)
{
    __shared__ float shmem[2];
    unsigned tid = threadIdx.x;
    if (tid == 0) { shmem[0] = 1.0001f; shmem[1] = 0.00001f; }
    __syncwarp();
    float b = shmem[0], c = shmem[1];
    float a0 = init_val + (float)tid * 0.001f;
    float a1 = a0 + 1.0f;

    unsigned long long t0, t1;

    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) { EMIT_CHAINS_2(b, c) }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) { EMIT_CHAINS_2(b, c) }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    float s = a0 + a1;
    if (__float_as_int(s) == 0x7FFFFFFF) out_result[tid] = s;
    if (tid == 0) out_timing[3] = t1 - t0;
}

// 4-chain ILP
__global__ __launch_bounds__(32, 1)
void kernel_ilp4(
    float* __restrict__ out_result,
    unsigned long long* __restrict__ out_timing,
    float init_val)
{
    __shared__ float shmem[2];
    unsigned tid = threadIdx.x;
    if (tid == 0) { shmem[0] = 1.0001f; shmem[1] = 0.00001f; }
    __syncwarp();
    float b = shmem[0], c = shmem[1];
    float a0 = init_val + (float)tid * 0.001f;
    float a1 = a0 + 1.0f, a2 = a0 + 2.0f, a3 = a0 + 3.0f;

    unsigned long long t0, t1;

    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) { EMIT_CHAINS_4(b, c) }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) { EMIT_CHAINS_4(b, c) }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    float s = a0 + a1 + a2 + a3;
    if (__float_as_int(s) == 0x7FFFFFFF) out_result[tid] = s;
    if (tid == 0) out_timing[4] = t1 - t0;
}

// 8-chain ILP
__global__ __launch_bounds__(32, 1)
void kernel_ilp8(
    float* __restrict__ out_result,
    unsigned long long* __restrict__ out_timing,
    float init_val)
{
    __shared__ float shmem[2];
    unsigned tid = threadIdx.x;
    if (tid == 0) { shmem[0] = 1.0001f; shmem[1] = 0.00001f; }
    __syncwarp();
    float b = shmem[0], c = shmem[1];
    float a0 = init_val + (float)tid * 0.001f;
    float a1 = a0 + 1.0f, a2 = a0 + 2.0f, a3 = a0 + 3.0f;
    float a4 = a0 + 4.0f, a5 = a0 + 5.0f, a6 = a0 + 6.0f, a7 = a0 + 7.0f;

    unsigned long long t0, t1;

    #pragma unroll 1
    for (int o = 0; o < 2; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) { EMIT_CHAINS_8(b, c) }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) : : "memory");
    #pragma unroll 1
    for (int o = 0; o < OUTER; o++) {
        #pragma unroll
        for (int i = 0; i < INNER; i++) { EMIT_CHAINS_8(b, c) }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) : : "memory");

    float s = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7;
    if (__float_as_int(s) == 0x7FFFFFFF) out_result[tid] = s;
    if (tid == 0) out_timing[5] = t1 - t0;
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char** argv)
{
    CHECK_CUDA(cudaSetDevice(0));

    cudaDeviceProp prop;
    CHECK_CUDA(cudaGetDeviceProperties(&prop, 0));
    printf("=== FFMA Latency Definitive Measurement ===\n");
    printf("Device: %s\n", prop.name);
    printf("SM compute capability: %d.%d\n", prop.major, prop.minor);

    printf("\n--- Current SM clock ---\n");
    fflush(stdout);
    system("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader 2>/dev/null | head -1");
    fflush(stdout);

    // Allocate output buffers
    int n_timings = 8;
    float* d_result;
    unsigned long long* d_timing;
    CHECK_CUDA(cudaMalloc(&d_result, 32 * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_timing, n_timings * sizeof(unsigned long long)));
    CHECK_CUDA(cudaMemset(d_timing, 0, n_timings * sizeof(unsigned long long)));

    float* h_result = (float*)malloc(32 * sizeof(float));
    unsigned long long* h_timing = (unsigned long long*)malloc(n_timings * sizeof(unsigned long long));

    // Warmup: run all kernels once to ensure GPU at boost clock
    printf("\nWarming up...\n");
    fflush(stdout);
    for (int rep = 0; rep < 3; rep++) {
        kernel_selfop<<<1, 32>>>(d_result, d_timing, 1.5f);
        kernel_self_const<<<1, 32>>>(d_result, d_timing, 1.5f);
        kernel_diffsrc<<<1, 32>>>(d_result, d_timing, 1.5f);
        kernel_ilp2<<<1, 32>>>(d_result, d_timing, 1.5f);
        kernel_ilp4<<<1, 32>>>(d_result, d_timing, 1.5f);
        kernel_ilp8<<<1, 32>>>(d_result, d_timing, 1.5f);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // Multiple runs to check stability
    int N_REPS = 5;
    unsigned long long timings_all[8][5] = {};

    printf("Running %d repetitions...\n", N_REPS);
    fflush(stdout);

    for (int rep = 0; rep < N_REPS; rep++) {
        CHECK_CUDA(cudaMemset(d_timing, 0, n_timings * sizeof(unsigned long long)));

        kernel_selfop<<<1, 32>>>(d_result, d_timing, 1.5f + rep * 0.001f);
        CHECK_CUDA(cudaDeviceSynchronize());
        kernel_self_const<<<1, 32>>>(d_result, d_timing, 1.5f + rep * 0.001f);
        CHECK_CUDA(cudaDeviceSynchronize());
        kernel_diffsrc<<<1, 32>>>(d_result, d_timing, 1.5f + rep * 0.001f);
        CHECK_CUDA(cudaDeviceSynchronize());
        kernel_ilp2<<<1, 32>>>(d_result, d_timing, 1.5f + rep * 0.001f);
        CHECK_CUDA(cudaDeviceSynchronize());
        kernel_ilp4<<<1, 32>>>(d_result, d_timing, 1.5f + rep * 0.001f);
        CHECK_CUDA(cudaDeviceSynchronize());
        kernel_ilp8<<<1, 32>>>(d_result, d_timing, 1.5f + rep * 0.001f);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUDA(cudaMemcpy(h_timing, d_timing, n_timings * sizeof(unsigned long long),
                              cudaMemcpyDeviceToHost));

        for (int i = 0; i < 6; i++) {
            timings_all[i][rep] = h_timing[i];
        }
    }

    // Compute medians
    // Simple insertion sort for 5 elements
    auto median5 = [](unsigned long long* arr) -> unsigned long long {
        unsigned long long a[5];
        for (int i = 0; i < 5; i++) a[i] = arr[i];
        // sort
        for (int i = 1; i < 5; i++) {
            unsigned long long key = a[i];
            int j = i - 1;
            while (j >= 0 && a[j] > key) { a[j+1] = a[j]; j--; }
            a[j+1] = key;
        }
        return a[2]; // median
    };

    unsigned long long med[6];
    for (int i = 0; i < 6; i++) med[i] = median5(timings_all[i]);

    long long total_ops_single = (long long)INNER * OUTER;  // total FMAs for single chain
    long long total_ops_2chain = total_ops_single * 2;
    long long total_ops_4chain = total_ops_single * 4;
    long long total_ops_8chain = total_ops_single * 8;

    printf("\n=== RAW TIMING DATA ===\n");
    printf("%-25s  %8s  %8s  %8s  %8s  %8s  (median)\n",
           "Variant", "rep0", "rep1", "rep2", "rep3", "rep4");
    const char* names[] = {
        "1-chain self-op", "1-chain self+const",
        "1-chain diffsrc",
        "2-chain ILP", "4-chain ILP", "8-chain ILP"
    };
    for (int i = 0; i < 6; i++) {
        printf("%-25s  %8llu  %8llu  %8llu  %8llu  %8llu  => %llu\n",
               names[i],
               timings_all[i][0], timings_all[i][1], timings_all[i][2],
               timings_all[i][3], timings_all[i][4],
               med[i]);
    }

    printf("\n=== CYCLES PER FMA ANALYSIS ===\n");
    printf("  Total FMAs per single-chain run: %lld\n", total_ops_single);
    printf("  (INNER=%d × OUTER=%d)\n", INNER, OUTER);
    printf("\n");

    // single-chain variants: cy/FMA = total_cycles / total_ops_single
    double cy_selfop    = (double)med[0] / total_ops_single;
    double cy_selfconst = (double)med[1] / total_ops_single;
    double cy_diffsrc   = (double)med[2] / total_ops_single;

    // ILP variants: total FMAs = total_ops_single * chains
    // cycles = med[3..5], cy/FMA = cycles / (total_ops_single * chains)
    // But for latency comparison, also report cy/chain-iter = cycles / total_ops_single
    // (since each chain independently cycles)
    double cy_ilp2      = (double)med[3] / total_ops_2chain;
    double cy_ilp4      = (double)med[4] / total_ops_4chain;
    double cy_ilp8      = (double)med[5] / total_ops_8chain;

    // cy/iter for latency: how many cycles does 1 dependent iteration take?
    // For 1 chain, this IS the latency.
    // For N chains, this = cy/FMA * N (each chain still has same latency)
    double lat_from_selfop    = cy_selfop;    // "latency" = cy/FMA for serial chain
    double lat_from_selfconst = cy_selfconst;
    double lat_from_diffsrc   = cy_diffsrc;

    // Throughput: cy/FMA for ILP=8 is throughput
    double tp_from_ilp8 = cy_ilp8;

    printf("SINGLE-CHAIN (latency-bound) — cy/FMA:\n");
    printf("  self-op  (fma a,a,a,0):    %.2f cy  [LIKELY INFLATED: self-dep port pressure]\n", cy_selfop);
    printf("  self+const (fma a,a,b,c):  %.2f cy  [cleaner: b,c from shmem]\n", cy_selfconst);
    printf("  diff-src (fma a,b,c,a):    %.2f cy  [cleanest: only accumulator dep]\n", cy_diffsrc);
    printf("\nILP CHAINS (throughput-approaching):\n");
    printf("  2-chain ILP:  %.2f cy/FMA  (chains hide %.1f cy latency)\n",
           cy_ilp2, (double)med[3] / total_ops_single);
    printf("  4-chain ILP:  %.2f cy/FMA  (chains hide %.1f cy latency)\n",
           cy_ilp4, (double)med[4] / total_ops_single);
    printf("  8-chain ILP:  %.2f cy/FMA  (chains hide %.1f cy latency)\n",
           cy_ilp8, (double)med[5] / total_ops_single);

    printf("\nKEY DEDUCTIONS:\n");
    printf("  Inflation factor (self-op vs self+const): %.2fx\n",
           cy_selfop / cy_selfconst);
    printf("  Inflation factor (self-op vs diff-src):   %.2fx\n",
           cy_selfop / cy_diffsrc);
    printf("  TRUE FFMA latency estimate (diff-src):    %.2f cy\n", lat_from_diffsrc);
    printf("  THROUGHPUT (8-chain ILP):                 %.2f cy/FMA\n", tp_from_ilp8);
    printf("  Throughput: ~%.1f FFMA/SM-cy (single warp, 4 partitions × %.2f)\n",
           4.0 / tp_from_ilp8, 1.0 / tp_from_ilp8);

    // Theoretical:
    // B300: 2 FMA pipes/SMSP, 4 SMSPs/SM → 8 FMAs/cy/SM at throughput
    // 1 FMA/cycle/pipe/SMSP → tp = 1 cy/FMA for fully-pipelined single chain
    printf("\nEXPECTED THEORETICAL BOUNDS:\n");
    printf("  True latency (pipelined FFMA): ~4 cy (matches Ampere/Hopper)\n");
    printf("  Max throughput (8+ chains):    <1 cy/FMA (dual-issue on B300)\n");

    printf("\n--- SM clock at end ---\n");
    fflush(stdout);
    system("nvidia-smi --query-gpu=clocks.current.sm --format=csv,noheader 2>/dev/null | head -1");
    fflush(stdout);

    cudaFree(d_result);
    cudaFree(d_timing);
    free(h_result);
    free(h_timing);

    return 0;
}
