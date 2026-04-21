// E7: sysmem atomics — atomicAdd on host-mapped memory vs device memory
#include <cuda_runtime.h>
#include <cstdio>

__global__ void atomic_kernel(unsigned int* target, int iters) {
    unsigned int sink = (unsigned)threadIdx.x;
    for (int i = 0; i < iters; i++) {
        sink ^= atomicAdd(target, 1u + sink);
    }
    if (sink == 0xDEADBEEF) target[1024] = sink;
}

int main() {
    int iters = 1000;
    int blocks = 32;
    int threads = 32;

    // Test 1: device memory atomic
    unsigned int* dev_mem;
    cudaMalloc(&dev_mem, 4096*sizeof(unsigned int));
    cudaMemset(dev_mem, 0, 4096*sizeof(unsigned int));

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    // Warmup
    atomic_kernel<<<blocks, threads>>>(dev_mem, iters);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    atomic_kernel<<<blocks, threads>>>(dev_mem, iters);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float dev_ms;
    cudaEventElapsedTime(&dev_ms, start, end);

    // Test 2: host-mapped memory atomic
    unsigned int* host_mem;
    cudaHostAlloc(&host_mem, 4096*sizeof(unsigned int), cudaHostAllocMapped);
    memset(host_mem, 0, 4096*sizeof(unsigned int));
    unsigned int* host_dev_ptr;
    cudaHostGetDevicePointer(&host_dev_ptr, host_mem, 0);

    atomic_kernel<<<blocks, threads>>>(host_dev_ptr, iters);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    atomic_kernel<<<blocks, threads>>>(host_dev_ptr, iters);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float sysmem_ms;
    cudaEventElapsedTime(&sysmem_ms, start, end);

    // Test 3: managed (unified) memory atomic
    unsigned int* mgd_mem;
    cudaMallocManaged(&mgd_mem, 4096*sizeof(unsigned int));
    memset(mgd_mem, 0, 4096*sizeof(unsigned int));

    atomic_kernel<<<blocks, threads>>>(mgd_mem, iters);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    atomic_kernel<<<blocks, threads>>>(mgd_mem, iters);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float mgd_ms;
    cudaEventElapsedTime(&mgd_ms, start, end);

    int total_atomics = blocks * threads * iters;
    printf("Device memory atomic: %.3f ms = %.2f Gatomic/s\n", dev_ms, total_atomics / dev_ms / 1e6);
    printf("Sysmem (host-mapped) atomic: %.3f ms = %.2f Gatomic/s (%.1fx slower)\n",
           sysmem_ms, total_atomics / sysmem_ms / 1e6, sysmem_ms / dev_ms);
    printf("Managed (unified) atomic: %.3f ms = %.2f Gatomic/s (%.1fx slower)\n",
           mgd_ms, total_atomics / mgd_ms / 1e6, mgd_ms / dev_ms);

    cudaFree(dev_mem);
    cudaFreeHost(host_mem);
    cudaFree(mgd_mem);
    return 0;
}
