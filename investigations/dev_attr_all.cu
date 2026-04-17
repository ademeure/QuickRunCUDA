// Probe ALL cudaDevAttr values on B300
#include <cuda_runtime.h>
#include <cstdio>

struct Attr { cudaDeviceAttr code; const char *name; };

#define A(name) {cudaDevAttr##name, #name}

int main() {
    cudaSetDevice(0);

    Attr attrs[] = {
        A(MaxThreadsPerBlock),
        A(MaxBlockDimX),
        A(MaxBlockDimY),
        A(MaxBlockDimZ),
        A(MaxGridDimX),
        A(MaxGridDimY),
        A(MaxGridDimZ),
        A(MaxSharedMemoryPerBlock),
        A(TotalConstantMemory),
        A(WarpSize),
        A(MaxPitch),
        A(MaxRegistersPerBlock),
        A(ClockRate),
        A(TextureAlignment),
        A(MultiProcessorCount),
        A(KernelExecTimeout),
        A(Integrated),
        A(CanMapHostMemory),
        A(ComputeMode),
        A(MaxTexture1DWidth),
        A(MaxTexture2DWidth),
        A(MaxTexture2DHeight),
        A(MaxTexture3DWidth),
        A(MaxTexture3DHeight),
        A(MaxTexture3DDepth),
        A(MaxTexture2DLayeredWidth),
        A(MaxTexture2DLayeredHeight),
        A(MaxTexture2DLayeredLayers),
        A(SurfaceAlignment),
        A(ConcurrentKernels),
        A(EccEnabled),
        A(PciBusId),
        A(PciDeviceId),
        A(TccDriver),
        A(MemoryClockRate),
        A(GlobalMemoryBusWidth),
        A(L2CacheSize),
        A(MaxThreadsPerMultiProcessor),
        A(AsyncEngineCount),
        A(UnifiedAddressing),
        A(MaxTexture1DLayeredWidth),
        A(MaxTexture1DLayeredLayers),
        A(MaxTexture2DGatherWidth),
        A(MaxTexture2DGatherHeight),
        A(MaxTexture3DWidthAlt),
        A(MaxTexture3DHeightAlt),
        A(MaxTexture3DDepthAlt),
        A(PciDomainId),
        A(TexturePitchAlignment),
        A(MaxTextureCubemapWidth),
        A(MaxTextureCubemapLayeredWidth),
        A(MaxTextureCubemapLayeredLayers),
        A(MaxSurface1DWidth),
        A(MaxSurface2DWidth),
        A(MaxSurface2DHeight),
        A(MaxSurface3DWidth),
        A(MaxSurface3DHeight),
        A(MaxSurface3DDepth),
        A(MaxSurface1DLayeredWidth),
        A(MaxSurface1DLayeredLayers),
        A(MaxSurface2DLayeredWidth),
        A(MaxSurface2DLayeredHeight),
        A(MaxSurface2DLayeredLayers),
        A(MaxSurfaceCubemapWidth),
        A(MaxSurfaceCubemapLayeredWidth),
        A(MaxSurfaceCubemapLayeredLayers),
        A(MaxTexture1DLinearWidth),
        A(MaxTexture2DLinearWidth),
        A(MaxTexture2DLinearHeight),
        A(MaxTexture2DLinearPitch),
        A(MaxTexture2DMipmappedWidth),
        A(MaxTexture2DMipmappedHeight),
        A(ComputeCapabilityMajor),
        A(ComputeCapabilityMinor),
        A(MaxTexture1DMipmappedWidth),
        A(StreamPrioritiesSupported),
        A(GlobalL1CacheSupported),
        A(LocalL1CacheSupported),
        A(MaxSharedMemoryPerMultiprocessor),
        A(MaxRegistersPerMultiprocessor),
        A(ManagedMemory),
        A(IsMultiGpuBoard),
        A(MultiGpuBoardGroupID),
        A(HostNativeAtomicSupported),
        A(SingleToDoublePrecisionPerfRatio),
        A(PageableMemoryAccess),
        A(ConcurrentManagedAccess),
        A(ComputePreemptionSupported),
        A(CanUseHostPointerForRegisteredMem),
        A(CooperativeLaunch),
        A(MaxSharedMemoryPerBlockOptin),
        A(CanFlushRemoteWrites),
        A(HostRegisterSupported),
        A(PageableMemoryAccessUsesHostPageTables),
        A(DirectManagedMemAccessFromHost),
        A(MaxBlocksPerMultiprocessor),
        A(MaxPersistingL2CacheSize),
        A(MaxAccessPolicyWindowSize),
        A(ReservedSharedMemoryPerBlock),
        A(SparseCudaArraySupported),
        A(HostRegisterReadOnlySupported),
        A(TimelineSemaphoreInteropSupported),
        A(MemoryPoolsSupported),
        A(GPUDirectRDMASupported),
        A(GPUDirectRDMAFlushWritesOptions),
        A(GPUDirectRDMAWritesOrdering),
        A(MemoryPoolSupportedHandleTypes),
        A(ClusterLaunch),
        A(DeferredMappingCudaArraySupported),
        A(IpcEventSupport),
        A(MemSyncDomainCount),
        A(NumaConfig),
        A(NumaId),
        A(MpsEnabled),
        A(HostNumaId),
    };

    printf("# B300 cudaDevAttr complete probe\n");
    printf("# %d attributes\n\n", (int)(sizeof(attrs)/sizeof(Attr)));

    int printed = 0;
    for (auto &a : attrs) {
        int v;
        cudaError_t r = cudaDeviceGetAttribute(&v, a.code, 0);
        if (r == cudaSuccess) {
            printf("  %-55s : %d\n", a.name, v);
            printed++;
        }
    }
    printf("\n# %d attributes successfully queried\n", printed);

    return 0;
}
