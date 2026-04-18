// Pipe contention matrix v2 — consistent ILP=4 chains across all tests
// Adds HMMA (tensor core) to confirm tensor pipe is independent of MIO
#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>

constexpr int N_INNER = 64;

// Each kernel has SAME structure: 4 independent chains, N_INNER inner iters,
// 1 op-per-chain per inner = 4 ops total per inner.
// Combo kernels add 4 STS per inner (one per chain).

__launch_bounds__(1024, 1) __global__ void k_sts(int *out, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            #pragma unroll
            for (int k = 0; k < 4; k++) {
                vsmem[slot + k * 1024] = v + i + j + k;
            }
        }
    }
}

__launch_bounds__(1024, 1) __global__ void k_ffma(int *out, int N) {
    float a = threadIdx.x * 1.001f, b = 0.999f, c = 0.001f;
    float d = threadIdx.x * 1.002f, e = 0.998f, f = 0.002f;
    float g = threadIdx.x * 1.003f, h = 0.997f, k = 0.003f;
    float m = threadIdx.x * 1.004f, n = 0.996f, p = 0.004f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            a = a * b + c;  d = d * e + f;  g = g * h + k;  m = m * n + p;
        }
    }
    if (a + d + g + m == 0.0f && N < 0) out[threadIdx.x] = 1;
}

__launch_bounds__(1024, 1) __global__ void k_iadd3(int *out, int N) {
    int a = threadIdx.x ^ 1, b = 1, c = 2;
    int d = threadIdx.x ^ 3, e = 5, f = 7;
    int g = threadIdx.x ^ 5, h = 11, k = 13;
    int m = threadIdx.x ^ 7, n = 17, p = 19;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            a = a + b + c;  d = d + e + f;  g = g + h + k;  m = m + n + p;
        }
    }
    // Anti-DCE: write at end
    out[blockIdx.x] = (N == 1234567) ? a + d + g + m : out[blockIdx.x];
}

__launch_bounds__(1024, 1) __global__ void k_mufu(int *out, int N) {
    float a = threadIdx.x * 0.001f + 1.5f;
    float b = a + 0.5f;
    float c = b + 0.5f;
    float d = c + 0.5f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            a = __frsqrt_rn(a + 1.0f);  b = __frsqrt_rn(b + 1.0f);
            c = __frsqrt_rn(c + 1.0f);  d = __frsqrt_rn(d + 1.0f);
        }
    }
    if (a + b + c + d == 0.0f && N < 0) out[threadIdx.x] = 1;
}

__launch_bounds__(1024, 1) __global__ void k_dfma(int *out, int N) {
    double a = threadIdx.x * 1.001, b = 0.999, c = 0.001;
    double d = threadIdx.x * 1.002, e = 0.998, f = 0.002;
    double g = threadIdx.x * 1.003, h = 0.997, k = 0.003;
    double m = threadIdx.x * 1.004, n = 0.996, p = 0.004;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            a = a * b + c;  d = d * e + f;  g = g * h + k;  m = m * n + p;
        }
    }
    if (a + d + g + m == 0.0 && N < 0) out[threadIdx.x] = 1;
}

// HMMA m16n8k16 BF16 via mma.sync (legacy tensor path)
__launch_bounds__(1024, 1) __global__ void k_hmma(int *out, int N) {
    using namespace nvcuda::wmma;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c;
    fill_fragment(a, __float2bfloat16(0.5f));
    fill_fragment(b, __float2bfloat16(0.7f));
    fill_fragment(c, 0.0f);
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_sync(c, a, b, c);
        }
    }
    if (c.x[0] == 0.0f && N < 0) out[threadIdx.x] = 1;
}

// Combos: each chain interleaves the op + STS

__launch_bounds__(1024, 1) __global__ void k_sts_ffma(int *out, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
    float a = threadIdx.x * 1.001f, b = 0.999f, c = 0.001f;
    float d = threadIdx.x * 1.002f, e = 0.998f, f = 0.002f;
    float g = threadIdx.x * 1.003f, h = 0.997f, k = 0.003f;
    float m = threadIdx.x * 1.004f, n = 0.996f, p = 0.004f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            vsmem[slot + 0*1024] = v+i+j; a = a * b + c;
            vsmem[slot + 1*1024] = v+i+j; d = d * e + f;
            vsmem[slot + 2*1024] = v+i+j; g = g * h + k;
            vsmem[slot + 3*1024] = v+i+j; m = m * n + p;
        }
    }
    if (a + d + g + m == 0.0f && N < 0) out[threadIdx.x] = 1;
}

__launch_bounds__(1024, 1) __global__ void k_sts_mufu(int *out, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
    float a = threadIdx.x * 0.001f + 1.5f;
    float b = a + 0.5f, c = b + 0.5f, d = c + 0.5f;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            vsmem[slot + 0*1024] = v+i+j; a = __frsqrt_rn(a + 1.0f);
            vsmem[slot + 1*1024] = v+i+j; b = __frsqrt_rn(b + 1.0f);
            vsmem[slot + 2*1024] = v+i+j; c = __frsqrt_rn(c + 1.0f);
            vsmem[slot + 3*1024] = v+i+j; d = __frsqrt_rn(d + 1.0f);
        }
    }
    if (a + b + c + d == 0.0f && N < 0) out[threadIdx.x] = 1;
}

__launch_bounds__(1024, 1) __global__ void k_sts_dfma(int *out, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
    double a = threadIdx.x * 1.001, b = 0.999, c = 0.001;
    double d = threadIdx.x * 1.002, e = 0.998, f = 0.002;
    double g = threadIdx.x * 1.003, h = 0.997, k = 0.003;
    double m = threadIdx.x * 1.004, n = 0.996, p = 0.004;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            vsmem[slot + 0*1024] = v+i+j; a = a * b + c;
            vsmem[slot + 1*1024] = v+i+j; d = d * e + f;
            vsmem[slot + 2*1024] = v+i+j; g = g * h + k;
            vsmem[slot + 3*1024] = v+i+j; m = m * n + p;
        }
    }
    if (a + d + g + m == 0.0 && N < 0) out[threadIdx.x] = 1;
}

__launch_bounds__(1024, 1) __global__ void k_sts_hmma(int *out, int N) {
    using namespace nvcuda::wmma;
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
    fragment<matrix_a, 16, 16, 16, __nv_bfloat16, row_major> a;
    fragment<matrix_b, 16, 16, 16, __nv_bfloat16, col_major> b;
    fragment<accumulator, 16, 16, 16, float> c;
    fill_fragment(a, __float2bfloat16(0.5f));
    fill_fragment(b, __float2bfloat16(0.7f));
    fill_fragment(c, 0.0f);
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            vsmem[slot + 0*1024] = v+i+j;
            mma_sync(c, a, b, c);
            vsmem[slot + 1*1024] = v+i+j;
            mma_sync(c, a, b, c);
            vsmem[slot + 2*1024] = v+i+j;
            mma_sync(c, a, b, c);
            vsmem[slot + 3*1024] = v+i+j;
            mma_sync(c, a, b, c);
        }
    }
    if (c.x[0] == 0.0f && N < 0) out[threadIdx.x] = 1;
}

template <typename Fn>
double bench(const char* name, Fn kfn, int *d_out, int N) {
    int blocks = 148, threads = 1024;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) kfn<<<blocks, threads>>>(d_out, N);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR %s: %s\n", name, cudaGetErrorString(cudaGetLastError())); return 0; }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        kfn<<<blocks, threads>>>(d_out, N);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    return best;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));
    int N = 200;
    printf("# Pipe matrix v2 (4 SMSPs, ILP=4 chains, identical structure for alone vs combo)\n\n");
    printf("# Each kernel: N_INNER × 4 ops per inner = same op count for X-alone vs X+STS combos\n\n");

    double t_sts   = bench("STS",   k_sts,   d_out, N);
    double t_ffma  = bench("FFMA",  k_ffma,  d_out, N);
    double t_iadd3 = bench("IADD3", k_iadd3, d_out, N);
    double t_mufu  = bench("MUFU",  k_mufu,  d_out, N);
    double t_dfma  = bench("DFMA",  k_dfma,  d_out, N);
    double t_hmma  = bench("HMMA",  k_hmma,  d_out, N);

    printf("# Alone times (ms):\n");
    printf("  STS:   %.4f\n", t_sts);
    printf("  FFMA:  %.4f\n", t_ffma);
    printf("  IADD3: %.4f\n", t_iadd3);
    printf("  MUFU:  %.4f\n", t_mufu);
    printf("  DFMA:  %.4f\n", t_dfma);
    printf("  HMMA:  %.4f\n", t_hmma);

    double t_sts_ffma = bench("X+STS", k_sts_ffma, d_out, N);
    double t_sts_mufu = bench("X+STS", k_sts_mufu, d_out, N);
    double t_sts_dfma = bench("X+STS", k_sts_dfma, d_out, N);
    double t_sts_hmma = bench("X+STS", k_sts_hmma, d_out, N);

    printf("\n# X+STS combo (4 X-ops + 4 STS-ops per inner):\n");
    auto report = [](const char* name, double tc, double ta, double tb) {
        double max = ta > tb ? ta : tb;
        double sum = ta + tb;
        const char* tag;
        if      (tc < max * 1.10) tag = "INDEPENDENT pipes";
        else if (tc < (max + sum) * 0.5) tag = "MOSTLY independent";
        else if (tc < sum * 0.95) tag = "PARTIAL overlap";
        else                      tag = "SHARED (serialized)";
        printf("  %-12s combo=%.4f  alone-X=%.4f  alone-STS=%.4f  max=%.4f  sum=%.4f → %s\n",
               name, tc, ta, tb, max, sum, tag);
    };
    report("STS+FFMA",  t_sts_ffma,  t_ffma,  t_sts);
    report("STS+MUFU",  t_sts_mufu,  t_mufu,  t_sts);
    report("STS+DFMA",  t_sts_dfma,  t_dfma,  t_sts);
    report("STS+HMMA",  t_sts_hmma,  t_hmma,  t_sts);

    return 0;
}
