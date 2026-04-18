// S1: Are 4 tensor cores per SM independent?
// Vary warps per SM and measure mma.sync throughput per warp + per SM

#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>

#define ITERS 5000
#define ILP 16  // try filling 10-cycle mma pipe with more chains

__launch_bounds__(WARPS_PER_BLK * 32, BLOCKS_PER_SM) __global__ void mma_kernel(float *out) {
    int laneId = threadIdx.x & 31;
    unsigned a0 = laneId, a1 = laneId+1, a2 = laneId+2, a3 = laneId+3;
    unsigned b0 = laneId*7, b1 = laneId*11;
    float c[ILP][4] = {0};
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int k = 0; k < ILP; k++) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[k][0]),"+f"(c[k][1]),"+f"(c[k][2]),"+f"(c[k][3])
                : "r"(a0),"r"(a1),"r"(a2),"r"(a3),"r"(b0),"r"(b1));
        }
    }
    float s = 0;
    for (int k = 0; k < ILP; k++) s += c[k][0]+c[k][1]+c[k][2]+c[k][3];
    if (s < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = s;
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    float *d_out; cudaMalloc(&d_out, 1<<24);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int blocks = 148 * BLOCKS_PER_SM;
    int threads = WARPS_PER_BLK * 32;
    int warps_per_sm = WARPS_PER_BLK * BLOCKS_PER_SM;

    for (int i = 0; i < 5; i++) mma_kernel<<<blocks, threads>>>(d_out);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(e0);
        mma_kernel<<<blocks, threads>>>(d_out);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    unsigned mhz; nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
    int total_warps = blocks * threads / 32;
    long mma = (long)total_warps * ITERS * ILP;
    long flops = mma * 16 * 8 * 16 * 2;
    double tflops = flops / (best/1000) / 1e12;
    double per_sm = tflops / 148;
    double per_warp = per_sm / warps_per_sm;
    printf("WARPS/SM=%2d: %.4f ms = %.0f TFLOPS aggregate, %.2f TFLOPS/SM, %.3f TFLOPS/warp, clk=%u\n",
           warps_per_sm, best, tflops, per_sm, per_warp, mhz);
    nvmlShutdown();
    return 0;
}
