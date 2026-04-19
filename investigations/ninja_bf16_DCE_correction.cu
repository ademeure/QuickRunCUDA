// BF16 mma.sync STRICT anti-DCE test — does mma.sync truly hit 90% spec?
// Use: thread-derived inputs + perturbation each iter + unconditional write
#include <cuda_runtime.h>
#include <cstdio>

constexpr int N_INNER = 64;

__device__ __forceinline__ void mma_bf16(unsigned (&d)[4], unsigned (&a)[4], unsigned (&b)[2], unsigned (&c)[4]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
        : "=r"(d[0]),"=r"(d[1]),"=r"(d[2]),"=r"(d[3])
        : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]),"r"(c[0]),"r"(c[1]),"r"(c[2]),"r"(c[3]));
}

// VARIANT 1: S1-style (constants, impossible-condition write — DCE-prone)
__launch_bounds__(512, 1) __global__ void k_s1(int *out, int N) {
    unsigned a0[4]={0x3f800001,0x3f800002,0x3f800003,0x3f800004};
    unsigned a1[4]={0x3f800005,0x3f800006,0x3f800007,0x3f800008};
    unsigned a2[4]={0x3f800009,0x3f80000a,0x3f80000b,0x3f80000c};
    unsigned a3[4]={0x3f80000d,0x3f80000e,0x3f80000f,0x3f800010};
    unsigned b0[2]={0x3f800001,0x3f800002};
    unsigned b1[2]={0x3f800003,0x3f800004};
    unsigned b2[2]={0x3f800005,0x3f800006};
    unsigned b3[2]={0x3f800007,0x3f800008};
    unsigned c0[4]={0},c1[4]={0},c2[4]={0},c3[4]={0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_bf16(c0, a0, b0, c0); mma_bf16(c1, a1, b1, c1);
            mma_bf16(c2, a2, b2, c2); mma_bf16(c3, a3, b3, c3);
        }
    }
    if (c0[0] == 0xDEADBEEFu && N < 0) out[threadIdx.x] = c0[0];  // impossible write
}

// VARIANT 2: STRICT anti-DCE
__launch_bounds__(512, 1) __global__ void k_strict(unsigned *out, int N) {
    unsigned a0[4]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002,threadIdx.x|0x3f800003,threadIdx.x|0x3f800004};
    unsigned a1[4]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006,threadIdx.x|0x3f800007,threadIdx.x|0x3f800008};
    unsigned a2[4]={threadIdx.x|0x3f800009,threadIdx.x|0x3f80000a,threadIdx.x|0x3f80000b,threadIdx.x|0x3f80000c};
    unsigned a3[4]={threadIdx.x|0x3f80000d,threadIdx.x|0x3f80000e,threadIdx.x|0x3f80000f,threadIdx.x|0x3f800010};
    unsigned b0[2]={threadIdx.x|0x3f800001,threadIdx.x|0x3f800002};
    unsigned b1[2]={threadIdx.x|0x3f800003,threadIdx.x|0x3f800004};
    unsigned b2[2]={threadIdx.x|0x3f800005,threadIdx.x|0x3f800006};
    unsigned b3[2]={threadIdx.x|0x3f800007,threadIdx.x|0x3f800008};
    unsigned c0[4]={0},c1[4]={0},c2[4]={0},c3[4]={0};
    for (int i = 0; i < N; i++) {
        #pragma unroll
        for (int j = 0; j < N_INNER; j++) {
            mma_bf16(c0, a0, b0, c0); mma_bf16(c1, a1, b1, c1);
            mma_bf16(c2, a2, b2, c2); mma_bf16(c3, a3, b3, c3);
        }
    }
    out[blockIdx.x * 512 + threadIdx.x] = c0[0]+c1[0]+c2[0]+c3[0];  // unconditional write
}

int main() {
    cudaSetDevice(0);
    int *d_outi; cudaMalloc(&d_outi, 148*512*sizeof(int));
    unsigned *d_outu; cudaMalloc(&d_outu, 148*512*sizeof(unsigned));
    int N = 200;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto bench = [&](const char* name, auto launch) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0); launch(); cudaEventRecord(e1); cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long warps = 148L * 16;
        long total_inst = warps * (long)N * N_INNER * 4;
        long ops_per_inst = 4096;
        double tflops = total_inst * ops_per_inst / (best/1000.0) / 1e12;
        printf("  %-25s  %.3f ms  %.1f TFLOPS  (%.1f%% of 2500 BF16 spec)\n",
               name, best, tflops, tflops/2500*100);
    };
    bench("V1: S1-style (DCE-prone)", [&](){k_s1<<<148, 512>>>(d_outi, N);});
    bench("V2: Strict anti-DCE",      [&](){k_strict<<<148, 512>>>(d_outu, N);});
    return 0;
}
