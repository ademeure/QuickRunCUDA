// V7 M4: Comprehensive cudaDeviceProp dump for B300
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    int n;
    cudaGetDeviceCount(&n);
    printf("CUDA devices: %d\n\n", n);

    cudaDeviceProp p;
    cudaGetDeviceProperties(&p, 0);

    printf("=== B300 Device 0 ===\n");
    printf("Name: %s\n", p.name);
    printf("CC: %d.%d (%s)\n", p.major, p.minor,
           p.major == 10 && p.minor == 3 ? "sm_103a / Blackwell SXM6" : "?");
    printf("\n--- Compute ---\n");
    printf("SMs: %d  SMSPs: 4 (per SM)\n", p.multiProcessorCount);
    printf("Max threads/block: %d  Max blocks: %dx%dx%d\n",
           p.maxThreadsPerBlock, p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
    printf("Max threads/SM: %d\n", p.maxThreadsPerMultiProcessor);
    printf("Warp size: %d\n", p.warpSize);
    printf("Concurrent kernels: %d\n", p.concurrentKernels);
    printf("\n--- Clocks ---\n");
    // Memory clock removed in CUDA 13 cudaDeviceProp; query via NVML
    printf("Memory bus width: %d bits\n", p.memoryBusWidth);
    printf("L2 cache: %d KB\n", p.l2CacheSize / 1024);
    printf("Persisting L2 max: %d KB\n", p.persistingL2CacheMaxSize / 1024);
    printf("\n--- Memory ---\n");
    printf("Global memory: %.1f GB\n", p.totalGlobalMem / 1e9);
    printf("Constant memory: %d KB\n", (int)(p.totalConstMem / 1024));
    printf("Shared memory/block: %d KB (max), %d KB (default)\n",
           (int)(p.sharedMemPerBlock / 1024), (int)(p.sharedMemPerBlockOptin / 1024));
    printf("Shared memory/SM: %d KB\n", (int)(p.sharedMemPerMultiprocessor / 1024));
    printf("Reg/block: %d  Reg/SM: %d\n", p.regsPerBlock, p.regsPerMultiprocessor);
    printf("\n--- Cooperation ---\n");
    printf("Cooperative launch: %s\n", p.cooperativeLaunch ? "yes" : "no");
    printf("Cooperative multi-device: REMOVED in CUDA 13\n");
    printf("Cluster supported: %s (max size 8 per V5 C5)\n",
           p.clusterLaunch ? "yes" : "no");
    printf("\n--- TMEM (Blackwell) ---\n");
    printf("(tcgen05 features not exposed via cudaDeviceProp; check PTX)\n");
    printf("\n--- ECC, RAS ---\n");
    printf("ECC: %s\n", p.ECCEnabled ? "enabled" : "disabled");
    printf("\n--- PCI / NVLink ---\n");
    printf("PCI domain:bus:dev = %d:%d:%d\n", p.pciDomainID, p.pciBusID, p.pciDeviceID);
    printf("MIG mode: %s\n", p.isMultiGpuBoard ? "multi-GPU board" : "single");

    return 0;
}
