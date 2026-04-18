// Simple time-bounded FFMA — direct fp32_peak_honest.cu pattern, no persistent
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>

#define ITERS 50000
#define ILP 16

extern "C" __global__ void k_peak(float *in_bc, float *out) {
    float b = in_bc[0];
    float c = in_bc[1];
    float a[ILP];
    #pragma unroll
    for (int j = 0; j < ILP; j++) a[j] = in_bc[2 + j] + threadIdx.x * 0.0001f;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll
        for (int j = 0; j < ILP; j++)
            asm volatile("fma.rn.f32 %0, %1, %2, %0;" : "+f"(a[j]) : "f"(b), "f"(c));
    }
    if (threadIdx.x == 0) {
        float s = 0;
        #pragma unroll
        for (int j = 0; j < ILP; j++) s += a[j];
        out[blockIdx.x] = s;
    }
}

int main() {
    cudaSetDevice(0);
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    int blocks = 148*2, threads = 1024;
    float h_bc[2 + ILP] = {1.00001f, 0.00001f};
    for (int i = 0; i < ILP; i++) h_bc[2+i] = 1.0f + i*0.1f;
    float *d_in, *d_out;
    cudaMalloc(&d_in, sizeof(h_bc));
    cudaMalloc(&d_out, blocks * sizeof(float));
    cudaMemcpy(d_in, h_bc, sizeof(h_bc), cudaMemcpyHostToDevice);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    for (int i = 0; i < 5; i++) k_peak<<<blocks, threads>>>(d_in, d_out);
    cudaDeviceSynchronize();
    float best = 1e30f;
    for (int i = 0; i < 20; i++) {
        cudaEventRecord(e0);
        k_peak<<<blocks, threads>>>(d_in, d_out);
        cudaEventRecord(e1); cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    unsigned mhz; nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
    long ffma = (long)blocks * threads * ITERS * ILP;
    double tflops = ffma * 2.0 / (best/1000) / 1e12;
    double th = 148*128*2*(mhz/1000.0)/1000;
    printf("# Proven recipe (1024 thr × 2 blk/SM, ILP=%d, ITERS=%d)\n", ILP, ITERS);
    printf("  Best: %.4f ms, %.2f TFLOPS = %.2f%% of %u MHz theoretical (%.1f)\n",
           best, tflops, tflops/th*100, mhz, th);
    nvmlShutdown();
    return 0;
}
