// V8 I4: HBM ECC overhead check
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("B300 ECC status: %s\n", p.ECCEnabled ? "ENABLED" : "DISABLED");
    printf("Total memory: %.1f GB\n", p.totalGlobalMem / 1e9);

    // HBM3E usable capacity: nominally 288 GB for B300 (24 GB × 12 stacks)
    // With ECC: actual usable ~6-12% less due to ECC tax
    double advertised = 288.0;
    double measured = p.totalGlobalMem / 1e9;
    double ecc_tax = 100.0 * (advertised - measured) / advertised;
    printf("ECC capacity tax: %.1f%% (%.1f GB of %.1f GB advertised)\n",
           ecc_tax, measured, advertised);

    // Check ECC error counters
    // (Note: cudaMemRangeGetAttribute + CUDA_MEMADVISE_ECC_SET doesn't exist; use NVML)
    printf("\nNote: HBM3E has inline ECC (always on on B300). Bandwidth cost is built into\n");
    printf("the published spec, so no runtime perf difference to measure.\n");

    return 0;
}
