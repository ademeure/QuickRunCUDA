// cudaPointerGetAttributes for various memory types
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>

const char* type_str(cudaMemoryType t) {
    switch (t) {
        case cudaMemoryTypeUnregistered: return "Unregistered";
        case cudaMemoryTypeHost: return "Host";
        case cudaMemoryTypeDevice: return "Device";
        case cudaMemoryTypeManaged: return "Managed";
        default: return "?";
    }
}

void check(const void *p, const char *name) {
    cudaPointerAttributes attr;
    cudaError_t err = cudaPointerGetAttributes(&attr, p);
    printf("  %-25s: ", name);
    if (err) {
        printf("ERR %s\n", cudaGetErrorString(err));
        return;
    }
    printf("type=%s  device=%d  host=%p  dev=%p\n",
           type_str(attr.type), attr.device, attr.hostPointer, attr.devicePointer);
}

int main() {
    cudaSetDevice(0);
    printf("# B300 cudaPointerGetAttributes for various pointer types\n\n");

    // Stack
    int stack_var = 42;
    check(&stack_var, "stack");

    // Heap (malloc)
    int *heap = (int*)malloc(sizeof(int));
    check(heap, "malloc");

    // Heap (aligned_alloc)
    int *aligned = (int*)aligned_alloc(4096, 4096);
    check(aligned, "aligned_alloc");

    // Pinned host (cudaMallocHost)
    int *pinned;
    cudaMallocHost(&pinned, sizeof(int));
    check(pinned, "cudaMallocHost");

    // HostAlloc with various flags
    int *ha_default;
    cudaHostAlloc(&ha_default, sizeof(int), cudaHostAllocDefault);
    check(ha_default, "HostAllocDefault");

    int *ha_mapped;
    cudaHostAlloc(&ha_mapped, sizeof(int), cudaHostAllocMapped);
    check(ha_mapped, "HostAllocMapped");

    int *ha_wc;
    cudaHostAlloc(&ha_wc, sizeof(int), cudaHostAllocWriteCombined);
    check(ha_wc, "HostAllocWC");

    // Registered memory
    int *reg_mem = (int*)aligned_alloc(4096, 4096);
    cudaHostRegister(reg_mem, 4096, cudaHostRegisterDefault);
    check(reg_mem, "Registered");
    cudaHostUnregister(reg_mem);

    // Device
    int *dev;
    cudaMalloc(&dev, sizeof(int));
    check(dev, "cudaMalloc");

    // Managed
    int *m;
    cudaMallocManaged(&m, sizeof(int));
    check(m, "cudaMallocManaged");

    // Async-pool device
    int *async_dev;
    cudaStream_t s; cudaStreamCreate(&s);
    cudaMallocAsync(&async_dev, sizeof(int), s);
    cudaStreamSynchronize(s);
    check(async_dev, "cudaMallocAsync");

    return 0;
}
