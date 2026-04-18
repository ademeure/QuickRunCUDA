// BF16 mma.sync with MULTIPLE accumulator chains to break RAW dependency
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>

#define ITERS 1000

__launch_bounds__(256, 4) __global__ void mma_8chains(float *out, int iters) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    // 8 independent accumulator chains
    float c[8][4] = {0};

    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        #pragma unroll
        for (int k = 0; k < 8; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float s = 0;
    for (int k = 0; k < 8; k++) s += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    if (s < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = s;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    float *d_out; cudaMalloc(&d_out, 1<<24);
    int blocks = 148*4, threads = 256;
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 5; i++) mma_8chains<<<blocks, threads>>>(d_out, ITERS);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        mma_8chains<<<blocks, threads>>>(d_out, ITERS);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    unsigned mhz; nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
    int warps = blocks * threads / 32;
    long total_mma = (long)warps * ITERS * 8;
    long total_flops = total_mma * 16 * 8 * 16 * 2;
    double tflops = total_flops / (best/1000) / 1e12;
    printf("# mma 8 chains, ILP=8: %.4f ms = %.0f TFLOPS, clk=%u MHz\n",
           best, tflops, mhz);

    nvmlShutdown();
    return 0;
}
