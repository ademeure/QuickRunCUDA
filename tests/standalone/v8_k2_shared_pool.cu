// V8 K2: Can cudaMemPool be shared across processes via IPC?
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cudaMemPool_t pool;
    cudaDeviceGetDefaultMemPool(&pool, 0);

    // Try to get IPC handle for the pool
    cudaMemPoolPtrExportData data;
    cudaError_t err = cudaMemPoolExportToShareableHandle(&data, pool,
                        cudaMemHandleTypePosixFileDescriptor, 0);
    if (err != cudaSuccess) {
        printf("MemPoolExportToShareableHandle: %s\n", cudaGetErrorString(err));
        printf("Default pool may not be shareable — need custom pool with handle type\n");
    } else {
        printf("Pool exported successfully\n");
    }

    // Create a POSIX-FD-shareable pool
    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.handleTypes = cudaMemHandleTypePosixFileDescriptor;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = 0;

    cudaMemPool_t custom_pool;
    err = cudaMemPoolCreate(&custom_pool, &props);
    if (err != cudaSuccess) {
        printf("Custom pool create: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Custom pool (POSIX FD shareable) created\n");

    // Export
    int fd;
    err = cudaMemPoolExportToShareableHandle(&fd, custom_pool,
                        cudaMemHandleTypePosixFileDescriptor, 0);
    if (err == cudaSuccess) {
        printf("Custom pool export OK, fd=%d\n", fd);
    } else {
        printf("Custom pool export: %s\n", cudaGetErrorString(err));
    }

    // Allocate from pool
    void* ptr;
    cudaStream_t s;
    cudaStreamCreate(&s);
    cudaMallocFromPoolAsync(&ptr, 1024*1024, custom_pool, s);
    cudaStreamSynchronize(s);
    printf("Allocated 1 MB from shareable pool at %p\n", ptr);

    return 0;
}
