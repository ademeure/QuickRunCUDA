// Try forcing low clock via NVML's nvmlDeviceSetGpuLockedClocks
#include <cuda_runtime.h>
#include <nvml.h>
#include <cstdio>

int main() {
    nvmlInit();
    nvmlDevice_t dev; nvmlDeviceGetHandleByIndex(0, &dev);
    unsigned mhz;
    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
    printf("Before: %u MHz\n", mhz);

    nvmlReturn_t r = nvmlDeviceSetGpuLockedClocks(dev, 1005, 1005);
    printf("nvmlDeviceSetGpuLockedClocks(1005,1005) = %d (%s)\n",
           (int)r, nvmlErrorString(r));

    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
    printf("After: %u MHz\n", mhz);

    // Now run a small FFMA workload and re-check
    cudaSetDevice(0);
    void *d; cudaMalloc(&d, 1<<20);
    cudaMemset(d, 0xab, 1<<20);
    cudaDeviceSynchronize();

    nvmlDeviceGetClockInfo(dev, NVML_CLOCK_SM, &mhz);
    printf("After memset: %u MHz\n", mhz);

    nvmlDeviceResetGpuLockedClocks(dev);
    nvmlShutdown();
    return 0;
}
