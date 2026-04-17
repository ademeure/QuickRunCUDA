// Power/temp/clock under sustained heavy compute
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>
#include <chrono>
#include <thread>

extern "C" __global__ void hot_kernel(float *out, int iters) {
    float a = threadIdx.x * 0.001f;
    float b = threadIdx.x * 0.002f;
    float c = threadIdx.x * 0.003f;
    float d = threadIdx.x * 0.004f;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f;
        b = b*1.0002f + 0.0002f;
        c = c*1.0003f + 0.0003f;
        d = d*1.0004f + 0.0004f;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    nvmlInit_v2();
    nvmlDevice_t dev;
    nvmlDeviceGetHandleByIndex_v2(0, &dev);

    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));
    cudaStream_t s; cudaStreamCreate(&s);

    // Bigger kernel: 10 ms each
    int iters = 1000000;

    printf("# B300 sustained hot loop power/clock/temp monitoring\n");
    printf("# Kernel: 148 × 256 thr × 1M iter (≈10 ms each)\n\n");
    printf("# %-6s %-12s %-10s %-10s %-10s %-12s\n",
           "t_s", "elapsed_ms", "SM_MHz", "Power_W", "Temp_C", "TFLOPS");

    auto t_start = std::chrono::high_resolution_clock::now();
    int batch = 0;
    while (true) {
        // Run a batch of kernels
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, s);
        for (int i = 0; i < 5; i++) hot_kernel<<<148, 256, 0, s>>>(d_out, iters);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms;
        cudaEventElapsedTime(&ms, e0, e1);

        long flops = (long)5 * 148 * 256 * iters * 4 * 2;
        double tflops = flops / (ms/1000.0) / 1e12;

        unsigned int sm_mhz, power_mw, temp_c;
        nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &sm_mhz);
        nvmlDeviceGetPowerUsage(dev, &power_mw);
        nvmlDeviceGetTemperature(dev, NVML_TEMPERATURE_GPU, &temp_c);

        auto t_now = std::chrono::high_resolution_clock::now();
        float t_s = std::chrono::duration<float>(t_now - t_start).count();
        printf("  %-6.1f %-12.2f %-10u %-10.1f %-10u %-12.1f\n",
               t_s, ms, sm_mhz, power_mw/1000.0, temp_c, tflops);

        cudaEventDestroy(e0); cudaEventDestroy(e1);

        batch++;
        if (t_s > 30) break;
    }

    nvmlShutdown();
    return 0;
}
