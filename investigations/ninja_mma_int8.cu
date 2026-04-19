// INT8 mma.sync m16n8k32 s8.s8.s32 — peak throughput
// Theoretical: same inst rate as BF16 m16n8k16, but K=32 instead of K=16
//   → 2× ops per inst → 2× TFLOPS
//   → spec: BF16 = 2500 TFLOPS, INT8 should be 5000 TOPS
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

__device__ __forceinline__ void mma_s8(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}

__launch_bounds__(512, 1) __global__ void k(unsigned *out, int N) {
    unsigned a0[4]={threadIdx.x|0x1,threadIdx.x|0x2,threadIdx.x|0x3,threadIdx.x|0x4};
    unsigned a1[4]={threadIdx.x|0x5,threadIdx.x|0x6,threadIdx.x|0x7,threadIdx.x|0x8};
    unsigned a2[4]={threadIdx.x|0x9,threadIdx.x|0xa,threadIdx.x|0xb,threadIdx.x|0xc};
    unsigned a3[4]={threadIdx.x|0xd,threadIdx.x|0xe,threadIdx.x|0xf,threadIdx.x|0x10};
    unsigned a4[4]={threadIdx.x|0x11,threadIdx.x|0x12,threadIdx.x|0x13,threadIdx.x|0x14};
    unsigned a5[4]={threadIdx.x|0x15,threadIdx.x|0x16,threadIdx.x|0x17,threadIdx.x|0x18};
    unsigned a6[4]={threadIdx.x|0x19,threadIdx.x|0x1a,threadIdx.x|0x1b,threadIdx.x|0x1c};
    unsigned a7[4]={threadIdx.x|0x1d,threadIdx.x|0x1e,threadIdx.x|0x1f,threadIdx.x|0x20};
    unsigned b0[2]={threadIdx.x|0x1,threadIdx.x|0x2};
    unsigned b1[2]={threadIdx.x|0x3,threadIdx.x|0x4};
    unsigned b2[2]={threadIdx.x|0x5,threadIdx.x|0x6};
    unsigned b3[2]={threadIdx.x|0x7,threadIdx.x|0x8};
    unsigned c0[4]={0},c1[4]={0},c2[4]={0},c3[4]={0};
    unsigned c4[4]={0},c5[4]={0},c6[4]={0},c7[4]={0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_s8(c0, a0, b0, c0);
            mma_s8(c1, a1, b1, c1);
            mma_s8(c2, a2, b2, c2);
            mma_s8(c3, a3, b3, c3);
            mma_s8(c4, a4, b0, c4);
            mma_s8(c5, a5, b1, c5);
            mma_s8(c6, a6, b2, c6);
            mma_s8(c7, a7, b3, c7);
        }
    }
    out[blockIdx.x * 512 + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0]+c4[0]+c5[0]+c6[0]+c7[0];
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
    long warps = 148L * 16, ops_per_inst = 2L * 16 * 8 * 32;  // 8192 ops/inst
    long total_ops = warps * (long)N * N_INNER * 8 * ops_per_inst;
    double tops = total_ops / (best/1000.0) / 1e12;
    printf("# INT8 mma.sync m16n8k32 s8.s8.s32 satfinite\n");
    printf("  best=%.3f ms  %.1f TOPS  (NVIDIA spec: 5000 INT8 dense, %.1f%%)\n",
           best, tops, tops/5000*100);
    return 0;
}
