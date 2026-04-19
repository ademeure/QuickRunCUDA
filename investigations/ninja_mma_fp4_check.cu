// What's the peak FP4 (e2m1) mma.sync throughput on B300 legacy tensor pipe?
//
// NVIDIA spec: B300 FP4 dense via tcgen05 = 15000 TFLOPS
// Question: what fraction can legacy mma.sync.m16n8k64.f4e2m1 achieve?
//
// PTX form: mma.sync.aligned.m16n8k64.row.col.f32.e2m1.e2m1.f32
//   - m=16, n=8, k=64 → 16*8*64*2 = 16384 FLOPS per inst
//   - 4× ops/inst vs BF16 m16n8k16 (4096 FLOPS)
//   - Should give ~4× BF16 TFLOPS if same inst rate
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

// Try FP4 e2m1
__device__ __forceinline__ void mma_fp4(unsigned (&d)[4],
                                        unsigned (&a)[4], unsigned (&b)[2],
                                        unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::f8f6f4.f32.e2m1.e2m1.f32 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
        : "=r"(d[0]), "=r"(d[1]), "=r"(d[2]), "=r"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]),
          "r"(c[0]), "r"(c[1]), "r"(c[2]), "r"(c[3]));
}

template<int THREADS>
__launch_bounds__(THREADS, 1) __global__ void k_mma_fp4(unsigned *out, int N) {
    unsigned a0[4] = {threadIdx.x | 0x40404040u, threadIdx.x | 0x41414141u, threadIdx.x | 0x42424242u, threadIdx.x | 0x43434343u};
    unsigned a1[4] = {threadIdx.x | 0x44444444u, threadIdx.x | 0x45454545u, threadIdx.x | 0x46464646u, threadIdx.x | 0x47474747u};
    unsigned a2[4] = {threadIdx.x | 0x48484848u, threadIdx.x | 0x49494949u, threadIdx.x | 0x4a4a4a4au, threadIdx.x | 0x4b4b4b4bu};
    unsigned a3[4] = {threadIdx.x | 0x4c4c4c4cu, threadIdx.x | 0x4d4d4d4du, threadIdx.x | 0x4e4e4e4eu, threadIdx.x | 0x4f4f4f4fu};
    unsigned b0[2] = {threadIdx.x | 0x40404040u, threadIdx.x | 0x41414141u};
    unsigned b1[2] = {threadIdx.x | 0x42424242u, threadIdx.x | 0x43434343u};
    unsigned b2[2] = {threadIdx.x | 0x44444444u, threadIdx.x | 0x45454545u};
    unsigned b3[2] = {threadIdx.x | 0x46464646u, threadIdx.x | 0x47474747u};
    unsigned c0[4] = {0,0,0,0}, c1[4] = {0,0,0,0}, c2[4] = {0,0,0,0}, c3[4] = {0,0,0,0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_fp4(c0, a0, b0, c0);
            mma_fp4(c1, a1, b1, c1);
            mma_fp4(c2, a2, b2, c2);
            mma_fp4(c3, a3, b3, c3);
        }
    }
    out[blockIdx.x * THREADS + threadIdx.x] = c0[0] + c1[0] + c2[0] + c3[0];
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 148 * 512 * sizeof(unsigned));
    int N = 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) k_mma_fp4<512><<<148, 512>>>(d_out, N);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0); k_mma_fp4<512><<<148, 512>>>(d_out, N); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long blocks = 148, threads = 512;
    long warps = blocks * (threads/32);  // 148*16 = 2368
    long ops_per_inst = 16384;  // m16n8k64 FP4
    long n_chains = 4;
    long total_inst = warps * (long)N * N_INNER * n_chains;
    long total_ops = total_inst * ops_per_inst;
    double tflops = total_ops / (best/1000.0) / 1e12;
    printf("# FP4 mma.sync.m16n8k64 e2m1 (kind::f8f6f4.block_scale)\n");
    printf("  best=%.3f ms  %.1f TFLOPS  (%.1f%% of 15000 spec)\n", best, tflops, tflops/15000*100);
    return 0;
}
