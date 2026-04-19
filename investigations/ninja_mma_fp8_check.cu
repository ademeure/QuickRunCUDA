// FP8 mma.sync.m16n8k32 e4m3.e4m3 — clean measurement (fix D4 DCE issue)
// Pattern matches working TF32 test: thread-derived inits + output sum
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

__device__ __forceinline__ void mma_fp8(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}

__launch_bounds__(512, 1) __global__ void k(unsigned *out, int N) {
    unsigned a0[4]={threadIdx.x|0x40404041,threadIdx.x|0x40404042,threadIdx.x|0x40404043,threadIdx.x|0x40404044};
    unsigned a1[4]={threadIdx.x|0x40404045,threadIdx.x|0x40404046,threadIdx.x|0x40404047,threadIdx.x|0x40404048};
    unsigned a2[4]={threadIdx.x|0x40404049,threadIdx.x|0x4040404a,threadIdx.x|0x4040404b,threadIdx.x|0x4040404c};
    unsigned a3[4]={threadIdx.x|0x4040404d,threadIdx.x|0x4040404e,threadIdx.x|0x4040404f,threadIdx.x|0x40404050};
    unsigned b0[2]={threadIdx.x|0x40404041,threadIdx.x|0x40404042};
    unsigned b1[2]={threadIdx.x|0x40404043,threadIdx.x|0x40404044};
    unsigned b2[2]={threadIdx.x|0x40404045,threadIdx.x|0x40404046};
    unsigned b3[2]={threadIdx.x|0x40404047,threadIdx.x|0x40404048};
    unsigned c0[4]={threadIdx.x|0x1,threadIdx.x|0x2,threadIdx.x|0x3,threadIdx.x|0x4};
    unsigned c1[4]={threadIdx.x|0x5,threadIdx.x|0x6,threadIdx.x|0x7,threadIdx.x|0x8};
    unsigned c2[4]={threadIdx.x|0x9,threadIdx.x|0xa,threadIdx.x|0xb,threadIdx.x|0xc};
    unsigned c3[4]={threadIdx.x|0xd,threadIdx.x|0xe,threadIdx.x|0xf,threadIdx.x|0x10};
    for (int i = 0; i < N; i++) {
        // perturb a slightly each iter to defeat DCE folding
        // use small values to avoid FP8 saturation
        a0[0] ^= (i << 16);
        a0[1] = a0[1] ^ ((i+1) << 16);
        b0[0] ^= (i << 24);
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_fp8(c0, a0, b0, c0);
            mma_fp8(c1, a1, b1, c1);
            mma_fp8(c2, a2, b2, c2);
            mma_fp8(c3, a3, b3, c3);
        }
    }
    out[blockIdx.x * 512 + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];
}

int main() {
    cudaSetDevice(0);
    unsigned *d_out; cudaMalloc(&d_out, 148 * 512 * sizeof(unsigned));
    int N = 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    for (int i = 0; i < 3; i++) k<<<148, 512>>>(d_out, N);
    cudaDeviceSynchronize();
    if (cudaPeekAtLastError() != cudaSuccess) { printf("ERR: %s\n", cudaGetErrorString(cudaGetLastError())); return 1; }
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0); k<<<148, 512>>>(d_out, N); cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long warps = 148L * 16, ops_per_inst = 2L * 16 * 8 * 32;  // 8192 ops
    long total_ops = warps * (long)N * N_INNER * 4 * ops_per_inst;
    double tflops = total_ops / (best/1000.0) / 1e12;
    printf("# FP8 mma.sync m16n8k32 e4m3.e4m3.f32\n");
    printf("  best=%.3f ms  %.1f TFLOPS  (NVIDIA spec: 5000, %.1f%%)\n",
           best, tflops, tflops/5000*100);
    return 0;
}
