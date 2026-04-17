// cudaDeviceGetLimit values
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);

    printf("# B300 cudaDeviceGetLimit values\n\n");

    auto get_limit = [](cudaLimit limit, const char *name) {
        size_t v = 0;
        cudaError_t err = cudaDeviceGetLimit(&v, limit);
        printf("  %-40s %zu B (%s)\n", name, v, cudaGetErrorString(err));
    };

    get_limit(cudaLimitStackSize, "StackSize (per thread)");
    get_limit(cudaLimitPrintfFifoSize, "PrintfFifoSize");
    get_limit(cudaLimitMallocHeapSize, "MallocHeapSize");
    get_limit(cudaLimitDevRuntimeSyncDepth, "DevRuntimeSyncDepth");
    get_limit(cudaLimitDevRuntimePendingLaunchCount, "DevRuntimePendingLaunchCount");
    get_limit(cudaLimitMaxL2FetchGranularity, "MaxL2FetchGranularity");
    get_limit(cudaLimitPersistingL2CacheSize, "PersistingL2CacheSize");

    return 0;
}
