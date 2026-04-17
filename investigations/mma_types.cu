// mma.sync different types comparison
#include <cuda_runtime.h>
#include <cstdio>

__global__ void mma_bf16(float *out, int iters) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    float c0=0,c1=0,c2=0,c3=0;
    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a0+1),"r"(a1+1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    if (c0+c1+c2+c3 < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3;
}

__global__ void mma_fp16(float *out, int iters) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    float c0=0,c1=0,c2=0,c3=0;
    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+f"(c0),"+f"(c1),"+f"(c2),"+f"(c3)
            : "r"(a0),"r"(a1),"r"(a0+1),"r"(a1+1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
    }
    if (c0+c1+c2+c3 < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3;
}

__global__ void mma_int8(int *out, int iters) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1;
    unsigned b0 = laneId*7;
    int c0=0,c1=0,c2=0,c3=0;
    for (int i = 0; i < iters; i++) {
        asm volatile(
            "mma.sync.aligned.m16n8k32.row.col.satfinite.s32.s8.s8.s32 "
            "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
            : "+r"(c0),"+r"(c1),"+r"(c2),"+r"(c3)
            : "r"(a0),"r"(a1),"r"(a0+1),"r"(a1+1),"r"(b0),"r"(b0+1));
    }
    if (c0+c1+c2+c3 < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = c0+c1+c2+c3;
}

int main() {
    cudaSetDevice(0);
    void *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters = 100000;
    int blocks = 148, threads = 256;

    auto bench = [&](auto launch, int flops_per_mma) {
        for (int i = 0; i < 3; i++) launch();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            launch();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        int warps = blocks * (threads/32);
        long total = (long)warps * iters * flops_per_mma;
        return std::pair<float, double>{best, total/(best/1000.0)/1e12};
    };

    printf("# B300 mma.sync type comparison (single chain per warp)\n\n");
    {
        auto [t, tflops] = bench([&]{ mma_bf16<<<blocks, threads>>>((float*)d_out, iters); }, 16*8*16*2);
        printf("  m16n8k16 BF16:  %.3f ms = %.0f TFLOPS\n", t, tflops);
    }
    {
        auto [t, tflops] = bench([&]{ mma_fp16<<<blocks, threads>>>((float*)d_out, iters); }, 16*8*16*2);
        printf("  m16n8k16 FP16:  %.3f ms = %.0f TFLOPS\n", t, tflops);
    }
    {
        auto [t, tflops] = bench([&]{ mma_int8<<<blocks, threads>>>((int*)d_out, iters); }, 16*8*32*2);
        printf("  m16n8k32 INT8:  %.3f ms = %.0f TOPS\n", t, tflops);
    }

    return 0;
}
