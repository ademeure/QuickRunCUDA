// Peak FFMA + power/clock simultaneously
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>

#define KERN(NAME) \
__launch_bounds__(256, 8) __global__ void NAME(float *out, int iters) { \
    float a0=threadIdx.x, a1=a0+1, a2=a0+2, a3=a0+3, a4=a0+4, a5=a0+5, a6=a0+6, a7=a0+7; \
    float b0=a0*2, b1=a1*2, b2=a2*2, b3=a3*2, b4=a4*2, b5=a5*2, b6=a6*2, b7=a7*2; \
    float c0=a0*3, c1=a1*3, c2=a2*3, c3=a3*3, c4=a4*3, c5=a5*3, c6=a6*3, c7=a7*3; \
    for (int i = 0; i < iters; i++) { \
        a0=a0*1.0001f+b0; b0=b0*1.0001f+c0; c0=c0*1.0001f+a0; \
        a1=a1*1.0001f+b1; b1=b1*1.0001f+c1; c1=c1*1.0001f+a1; \
        a2=a2*1.0001f+b2; b2=b2*1.0001f+c2; c2=c2*1.0001f+a2; \
        a3=a3*1.0001f+b3; b3=b3*1.0001f+c3; c3=c3*1.0001f+a3; \
        a4=a4*1.0001f+b4; b4=b4*1.0001f+c4; c4=c4*1.0001f+a4; \
        a5=a5*1.0001f+b5; b5=b5*1.0001f+c5; c5=c5*1.0001f+a5; \
        a6=a6*1.0001f+b6; b6=b6*1.0001f+c6; c6=c6*1.0001f+a6; \
        a7=a7*1.0001f+b7; b7=b7*1.0001f+c7; c7=c7*1.0001f+a7; \
    } \
    float s=a0+a1+a2+a3+a4+a5+a6+a7+b0+b1+b2+b3+b4+b5+b6+b7+c0+c1+c2+c3+c4+c5+c6+c7; \
    if (s < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = s; \
}

KERN(peak_ffma)

int main() {
    cudaSetDevice(0);
    nvmlInit_v2();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex_v2(0, &dev);

    float *d_out; cudaMalloc(&d_out, 148 * 1024 * sizeof(float));
    cudaStream_t s; cudaStreamCreate(&s);

    int iters = 1000000;  // ~13 ms each kernel
    int blocks = 148, threads = 256;

    printf("# B300 sustained peak FFMA + power monitoring (15s)\n");
    printf("# 256 thr × 24 in-flight FMA × 1M iter\n\n");
    printf("# %-8s %-12s %-10s %-10s %-12s\n",
           "t_s", "elapsed_ms", "Power_W", "Temp_C", "TFLOPS");

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    auto t_start = std::chrono::high_resolution_clock::now();
    while (true) {
        cudaEventRecord(e0, s);
        peak_ffma<<<blocks, threads, 0, s>>>(d_out, iters);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);

        long ops = (long)blocks * threads * iters * 24 * 2;
        double tflops = ops / (ms/1000.0) / 1e12;

        unsigned int pw, t_c;
        nvmlDeviceGetPowerUsage(dev, &pw);
        nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &t_c);

        auto t_now = std::chrono::high_resolution_clock::now();
        float t_s = std::chrono::duration<float>(t_now - t_start).count();
        printf("  %-8.1f %-12.2f %-10.1f %-10u %-12.1f\n", t_s, ms, pw/1000.0, t_c, tflops);

        if (t_s > 12) break;
    }

    nvmlShutdown();
    return 0;
}
