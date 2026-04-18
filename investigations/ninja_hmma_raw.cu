// Raw mma.sync (no wmma fragment overhead) + STS contention
// To isolate whether HMMA anomaly is wmma-specific or true tensor+MIO contention.
//
// Use mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 directly.
// 4 chains of HMMA in registers.
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>

constexpr int N_INNER = 64;

// HMMA via inline PTX (pure register, no SMEM)
__device__ __forceinline__ void mma_b16(unsigned (&d)[4],
                                        unsigned (&a)[4], unsigned (&b)[2],
                                        unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

__launch_bounds__(1024, 1) __global__ void k_hmma(int *out, int N) {
    unsigned a0[4] = {0x3f800001, 0x3f800002, 0x3f800003, 0x3f800004};
    unsigned a1[4] = {0x3f800005, 0x3f800006, 0x3f800007, 0x3f800008};
    unsigned a2[4] = {0x3f800009, 0x3f80000a, 0x3f80000b, 0x3f80000c};
    unsigned a3[4] = {0x3f80000d, 0x3f80000e, 0x3f80000f, 0x3f800010};
    unsigned b0[2] = {0x3f800001, 0x3f800002};
    unsigned b1[2] = {0x3f800003, 0x3f800004};
    unsigned b2[2] = {0x3f800005, 0x3f800006};
    unsigned b3[2] = {0x3f800007, 0x3f800008};
    unsigned c0[4] = {0, 0, 0, 0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_b16(c0, a0, b0, c0);
            mma_b16(c1, a1, b1, c1);
            mma_b16(c2, a2, b2, c2);
            mma_b16(c3, a3, b3, c3);
        }
    }
    if (c0[0] == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0];
}

__launch_bounds__(1024, 1) __global__ void k_sts(int *out, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            #pragma unroll
            for (int k = 0; k < 4; k++) vsmem[slot + k * 1024] = v + i + j + k;
        }
    }
}

__launch_bounds__(1024, 1) __global__ void k_hmma_sts(int *out, int N) {
    __shared__ int smem[1024 * 4];
    volatile int *vsmem = smem;
    int slot = (threadIdx.x >> 5) * 32 + (threadIdx.x & 31);
    int v = threadIdx.x;
    unsigned a0[4] = {0x3f800001, 0x3f800002, 0x3f800003, 0x3f800004};
    unsigned a1[4] = {0x3f800005, 0x3f800006, 0x3f800007, 0x3f800008};
    unsigned a2[4] = {0x3f800009, 0x3f80000a, 0x3f80000b, 0x3f80000c};
    unsigned a3[4] = {0x3f80000d, 0x3f80000e, 0x3f80000f, 0x3f800010};
    unsigned b0[2] = {0x3f800001, 0x3f800002};
    unsigned b1[2] = {0x3f800003, 0x3f800004};
    unsigned b2[2] = {0x3f800005, 0x3f800006};
    unsigned b3[2] = {0x3f800007, 0x3f800008};
    unsigned c0[4] = {0, 0, 0, 0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            vsmem[slot + 0*1024] = v+i+j; mma_b16(c0, a0, b0, c0);
            vsmem[slot + 1*1024] = v+i+j; mma_b16(c1, a1, b1, c1);
            vsmem[slot + 2*1024] = v+i+j; mma_b16(c2, a2, b2, c2);
            vsmem[slot + 3*1024] = v+i+j; mma_b16(c3, a3, b3, c3);
        }
    }
    if (c0[0] == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0];
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
    double t_hmma = bench("HMMA-raw",     k_hmma,     d_out, N);
    double t_sts  = bench("STS",          k_sts,      d_out, N);
    double t_combo = bench("HMMA-raw+STS", k_hmma_sts, d_out, N);
    printf("# Raw mma.sync (no wmma) HMMA + STS\n");
    printf("  HMMA-raw alone:   %.4f ms\n", t_hmma);
    printf("  STS alone:        %.4f ms\n", t_sts);
    printf("  HMMA-raw+STS:     %.4f ms\n", t_combo);
    printf("  max-of-alone:     %.4f ms\n", t_hmma > t_sts ? t_hmma : t_sts);
    printf("  sum-of-alone:     %.4f ms\n", t_hmma + t_sts);
    return 0;
}
