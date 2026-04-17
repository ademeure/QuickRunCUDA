// mma.sync with ILP=2 (two independent mma per warp)
#include <cuda_runtime.h>
#include <cstdio>

__global__ void mma_ilp2(float *out, int iters) {
    int laneId = threadIdx.x & 31;

    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    unsigned a4 = laneId*2, a5 = laneId*2+1, a6 = laneId*2+2, a7 = laneId*2+3;
    unsigned b2 = laneId*13, b3 = laneId*17;

    float c0=0,c1=0,c2=0,c3=0;
    float d0=0,d1=0,d2=0,d3=0;

    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0), "+f"(c1), "+f"(c2), "+f"(c3)
            : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(d0), "+f"(d1), "+f"(d2), "+f"(d3)
            : "r"(a4), "r"(a5), "r"(a6), "r"(a7), "r"(b2), "r"(b3));
    }
    if (c0+c1+c2+c3+d0+d1+d2+d3 < -1e30f)
        out[blockIdx.x * blockDim.x + threadIdx.x] = c0+c1+c2+c3+d0+d1+d2+d3;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 256;

    for (int i = 0; i < 3; i++) mma_ilp2<<<blocks, threads>>>(d_out, iters);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(e0);
        mma_ilp2<<<blocks, threads>>>(d_out, iters);
        cudaEventRecord(e1);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }

    int warps = blocks * (threads / 32);
    long total_mma = (long)warps * iters * 2;  // 2 ILP
    long total_flops = total_mma * 4096;
    double tflops = total_flops / (best/1000.0) / 1e12;
    printf("# B300 mma.sync m16n8k16 BF16 with ILP=2\n");
    printf("  Time: %.3f ms\n  TFLOPS: %.0f\n", best, tflops);

    return 0;
}
