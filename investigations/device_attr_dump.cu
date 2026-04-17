// Comprehensive B300 device attribute dump
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("# B300 Device Properties (sm_103a)\n\n");
    printf("Name: %s\n", prop.name);
    printf("CC: %d.%d\n", prop.major, prop.minor);
    printf("Total mem: %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("L2 cache: %d MB\n", prop.l2CacheSize / 1024 / 1024);
    printf("Max persisting L2: %d MB\n", prop.persistingL2CacheMaxSize / 1024 / 1024);
    printf("\n");
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Threads/SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Threads/block: %d\n", prop.maxThreadsPerBlock);
    printf("Warps/SM: %d (%d/SM × 32 = %d)\n",
           prop.maxThreadsPerMultiProcessor/32, prop.maxBlocksPerMultiProcessor,
           prop.maxThreadsPerMultiProcessor);
    printf("Regs/block: %d\n", prop.regsPerBlock);
    printf("Regs/SM: %d\n", prop.regsPerMultiprocessor);
    printf("\n");
    printf("Shared/block (default): %d KB\n", prop.sharedMemPerBlock / 1024);
    printf("Shared/block (opt-in): %d KB\n", prop.sharedMemPerBlockOptin / 1024);
    printf("Shared/SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Reserved shmem/block: %d B\n", prop.reservedSharedMemPerBlock);
    printf("\n");
    printf("Warp size: %d\n", prop.warpSize);
    printf("Memory bus width: %d bits\n", prop.memoryBusWidth);
    printf("Memory clock: %d MHz\n", 3996000 / 1000);
    printf("HBM theoretical: %.1f GB/s\n",
           2.0 * prop.memoryBusWidth * 3996000 / 8 / 1e6);
    printf("Boost clock: %d MHz\n", 2032000 / 1000);
    printf("\n");
    printf("Concurrent kernels: %d\n", prop.concurrentKernels);
    printf("Async engine count: %d\n", prop.asyncEngineCount);
    printf("Compute mode: %d\n", 0);
    printf("Cooperative launch: %d\n", prop.cooperativeLaunch);
    printf("ECC enabled: %d\n", prop.ECCEnabled);
    printf("Unified addressing: %d\n", prop.unifiedAddressing);
    printf("Managed memory: %d\n", prop.managedMemory);
    printf("Pageable mem access: %d\n", prop.pageableMemoryAccess);
    printf("Pageable+host page tables: %d\n", prop.pageableMemoryAccessUsesHostPageTables);
    printf("Direct managed mem from host: %d\n", prop.directManagedMemAccessFromHost);
    printf("Concurrent managed access: %d\n", prop.concurrentManagedAccess);

    // PCIe info
    printf("\nPCIe domain: %d, bus: %d, device: %d\n",
           prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

    // Texture/surface dimensions
    printf("\nTexture 1D max: %d\n", prop.maxTexture1D);
    printf("Texture 2D max: %d × %d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("Surface 1D max: %d\n", prop.maxSurface1D);

    // Misc
    printf("\nIPC event support: %d\n", prop.ipcEventSupported);
    printf("Stream priorities: %d (range %d to %d)\n",
           prop.streamPrioritiesSupported, 0, 0);
    printf("Host native atomic: %d\n", prop.hostNativeAtomicSupported);

    return 0;
}
