// cudaMemRangeGetAttribute on various allocations
#include <cuda_runtime.h>
#include <cstdio>

void check_managed(void *p, size_t bytes, const char *name) {
    int read_only = -1, preferred_loc = -1, last_prefetch = -1;
    cudaError_t err1 = cudaMemRangeGetAttribute(&read_only, sizeof(int),
        cudaMemRangeAttributeReadMostly, p, bytes);
    cudaError_t err2 = cudaMemRangeGetAttribute(&preferred_loc, sizeof(int),
        cudaMemRangeAttributePreferredLocation, p, bytes);
    cudaError_t err3 = cudaMemRangeGetAttribute(&last_prefetch, sizeof(int),
        cudaMemRangeAttributeLastPrefetchLocation, p, bytes);

    printf("  %-30s ReadMostly=%d PrefLoc=%d LastPrefetch=%d\n",
           name, read_only, preferred_loc, last_prefetch);
}

int main() {
    cudaSetDevice(0);

    printf("# B300 cudaMemRangeGetAttribute on managed memory\n\n");

    size_t bytes = 64 * 1024;
    void *p; cudaMallocManaged(&p, bytes);
    check_managed(p, bytes, "Managed default");

    cudaMemLocation loc = {cudaMemLocationTypeDevice, 0};
    cudaMemAdvise(p, bytes, cudaMemAdviseSetReadMostly, loc);
    check_managed(p, bytes, "After SetReadMostly");

    cudaMemAdvise(p, bytes, cudaMemAdviseSetPreferredLocation, loc);
    check_managed(p, bytes, "After SetPreferredLocation=GPU");

    cudaMemPrefetchAsync(p, bytes, loc, 0, 0);
    cudaDeviceSynchronize();
    check_managed(p, bytes, "After prefetch to GPU");

    cudaMemLocation cpu_loc = {cudaMemLocationTypeHost, 0};
    cudaMemPrefetchAsync(p, bytes, cpu_loc, 0, 0);
    cudaDeviceSynchronize();
    check_managed(p, bytes, "After prefetch to CPU");

    cudaFree(p);
    return 0;
}
