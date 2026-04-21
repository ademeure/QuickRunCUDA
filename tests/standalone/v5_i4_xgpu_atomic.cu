// V5 I4: Cross-GPU atomic via NVLink (P2P)
// Compare:
//   1. Local atomic (same GPU)
//   2. Cross-GPU atomic to peer GPU memory via P2P
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
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);

    unsigned int* dev0_mem;
    cudaMalloc(&dev0_mem, 4096);
    cudaMemset(dev0_mem, 0, 4096);

    cudaSetDevice(1);
    unsigned int* dev1_mem;
    cudaMalloc(&dev1_mem, 4096);
    cudaMemset(dev1_mem, 0, 4096);

    int iters = 1000;
    int blocks = 32, threads = 32;

    cudaEvent_t s, e;
    cudaSetDevice(0);
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // Warmup
    atomic_kernel<<<blocks, threads>>>(dev0_mem, iters);
    cudaDeviceSynchronize();

    // Test 1: local atomic on dev0
    cudaEventRecord(s);
    atomic_kernel<<<blocks, threads>>>(dev0_mem, iters);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float local_ms;
    cudaEventElapsedTime(&local_ms, s, e);

    // Test 2: cross-GPU atomic from dev0 → dev1 mem (via NVLink)
    atomic_kernel<<<blocks, threads>>>(dev1_mem, iters); // warmup
    cudaDeviceSynchronize();
    cudaEventRecord(s);
    atomic_kernel<<<blocks, threads>>>(dev1_mem, iters);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float xgpu_ms;
    cudaEventElapsedTime(&xgpu_ms, s, e);

    int total_atomics = blocks * threads * iters;
    printf("Local atomic (same GPU):     %.3f ms = %.2f Gatomic/s\n",
           local_ms, total_atomics / local_ms / 1e6);
    printf("Cross-GPU atomic (NVLink):   %.3f ms = %.2f Gatomic/s\n",
           xgpu_ms, total_atomics / xgpu_ms / 1e6);
    printf("Cross-GPU slowdown: %.1fx\n", xgpu_ms / local_ms);

    return 0;
}
