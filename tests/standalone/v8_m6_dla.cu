// V8 M6: Check if B300 has DLA (Deep Learning Accelerator) like Jetson
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int n;
    cudaGetDeviceCount(&n);
    cudaDeviceProp p;

    for (int i = 0; i < n; i++) {
        cudaGetDeviceProperties(&p, i);
        printf("Device %d: %s (CC %d.%d)\n", i, p.name, p.major, p.minor);
        // DLA-related fields would be in the SoC line
        printf("  isMultiGpuBoard: %d\n", p.isMultiGpuBoard);
        printf("  integrated: %d (1 = integrated SoC like Jetson)\n", p.integrated);
        printf("  asyncEngineCount: %d\n", p.asyncEngineCount);
        printf("  unifiedAddressing: %d\n", p.unifiedAddressing);
        // No direct DLA field; check via cudaDeviceProp
    }
    printf("\nNote: DLA is a Jetson/SoC feature; B300 SXM6 is data center GPU.\n");
    printf("DLA would only appear on Tegra/Jetson platforms, not on B300.\n");

    return 0;
}
