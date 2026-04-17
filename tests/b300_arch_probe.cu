// Comprehensive B300 architectural probe
#include <cuda_runtime.h>
#include <cstdio>

#define A(name) cudaDeviceGetAttribute(&v, cudaDevAttr##name, 0); printf("  %-50s : %d\n", #name, v)

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("# B300 device properties (full)\n");
    printf("  Name: %s\n", prop.name);
    printf("  PCI Bus ID: %s\n", prop.pciBusID ? "present" : "?");
    printf("  Compute Cap: %d.%d\n", prop.major, prop.minor);
    printf("  SMs: %d\n", prop.multiProcessorCount);
    int clk_attr;
    cudaDeviceGetAttribute(&clk_attr, cudaDevAttrClockRate, 0);
    printf("  Clock rate: %d MHz\n", clk_attr / 1000);
    cudaDeviceGetAttribute(&clk_attr, cudaDevAttrMemoryClockRate, 0);
    printf("  Memory Clock: %d MHz\n", clk_attr / 1000);
    cudaDeviceGetAttribute(&clk_attr, cudaDevAttrGlobalMemoryBusWidth, 0);
    printf("  Memory bus width: %d bits\n", clk_attr);
    printf("  Total global mem: %.1f GB\n", prop.totalGlobalMem / (1024.f*1024.f*1024.f));
    printf("  L2 Cache Size: %d bytes (%.1f MB)\n", prop.l2CacheSize, prop.l2CacheSize/(1024.f*1024.f));
    printf("  Shared mem per block: %zu (%.1f KB)\n", prop.sharedMemPerBlock, prop.sharedMemPerBlock/1024.f);
    printf("  Shared mem per SM: %zu (%.1f KB)\n", prop.sharedMemPerMultiprocessor, prop.sharedMemPerMultiprocessor/1024.f);
    printf("  Reserved shared/block: %zu\n", prop.reservedSharedMemPerBlock);
    printf("  Shared mem per block opt-in: %zu (%.1f KB)\n", prop.sharedMemPerBlockOptin, prop.sharedMemPerBlockOptin/1024.f);
    printf("  Regs per block: %d\n", prop.regsPerBlock);
    printf("  Regs per SM: %d\n", prop.regsPerMultiprocessor);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Max blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);
    printf("  Warp size: %d\n", prop.warpSize);
    printf("  ECC enabled: %d\n", prop.ECCEnabled);
    printf("  TCC mode: %d\n", prop.tccDriver);
    printf("  Async engines: %d\n", prop.asyncEngineCount);
    printf("  Concurrent kernels: %d\n", prop.concurrentKernels);
    printf("  Concurrent managed access: %d\n", prop.concurrentManagedAccess);
    printf("  Cooperative launch: %d\n", prop.cooperativeLaunch);
    printf("  Stream priorities supported: %d\n", prop.streamPrioritiesSupported);
    printf("  Cluster launch: %d\n", prop.clusterLaunch);
    printf("  Memory pools supported: %d\n", prop.memoryPoolsSupported);
    printf("  Unified function ptrs: %d\n", prop.unifiedFunctionPointers);
    printf("  GPU Direct RDMA: %d\n", prop.gpuDirectRDMASupported);
    printf("  Sparse CUDA arrays: %d\n", prop.sparseCudaArraySupported);
    printf("  Device NUMA config: %d\n", prop.deviceNumaConfig);
    printf("  Device NUMA ID: %d\n", prop.deviceNumaId);

    int v;
    printf("\n# Specific cudaDevAttr probes\n");
    A(MultiProcessorCount);
    A(MaxThreadsPerBlock);
    A(MaxBlocksPerMultiprocessor);
    A(MaxThreadsPerMultiProcessor);
    A(WarpSize);
    A(MaxRegistersPerBlock);
    A(MaxRegistersPerMultiprocessor);
    A(MaxSharedMemoryPerBlock);
    A(MaxSharedMemoryPerBlockOptin);
    A(MaxSharedMemoryPerMultiprocessor);
    A(ReservedSharedMemoryPerBlock);
    A(L2CacheSize);
    A(MaxPersistingL2CacheSize);
    A(MaxAccessPolicyWindowSize);
    A(MaxBlockDimX);
    A(MaxBlockDimY);
    A(MaxBlockDimZ);
    A(MaxGridDimX);
    A(MaxGridDimY);
    A(MaxGridDimZ);
    A(ClockRate);
    A(MemoryClockRate);
    A(GlobalMemoryBusWidth);
    A(GlobalL1CacheSupported);
    A(LocalL1CacheSupported);
    A(MultiGpuBoardGroupID);
    A(SingleToDoublePrecisionPerfRatio);
    A(PageableMemoryAccess);
    A(ConcurrentManagedAccess);
    A(ComputePreemptionSupported);
    A(CooperativeLaunch);
    A(MemSyncDomainCount);
    A(IpcEventSupport);
    A(MemoryPoolsSupported);
    A(GPUDirectRDMASupported);
    A(SparseCudaArraySupported);
    A(ClusterLaunch);
    A(MaxBlocksPerMultiprocessor);
    A(NumaConfig);
    A(NumaId);
    A(MpsEnabled);
    A(HostNumaId);

    return 0;
}
