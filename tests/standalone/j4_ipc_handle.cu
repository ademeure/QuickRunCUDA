// J4: cudaIpcMemHandle — cross-process sharing of GPU memory
// Test: process A allocates, gets handle. Process B opens handle, accesses memory.
// Single-process test: same-process IPC handle round-trip + access cost
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void writer(int* p, int val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *p = val;
}

__global__ void reader(int* p, int* out) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out = *p;
}

int main() {
    cudaSetDevice(0);

    int* dev_a;
    cudaMalloc(&dev_a, sizeof(int));

    // Get IPC handle
    cudaIpcMemHandle_t handle;
    cudaError_t err = cudaIpcGetMemHandle(&handle, dev_a);
    if (err != cudaSuccess) {
        printf("cudaIpcGetMemHandle failed: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("cudaIpcGetMemHandle succeeded, handle size = %zu bytes\n", sizeof(handle));

    // Open handle (same process — should work but unusual)
    void* dev_b;
    err = cudaIpcOpenMemHandle(&dev_b, handle, cudaIpcMemLazyEnablePeerAccess);
    if (err != cudaSuccess) {
        printf("cudaIpcOpenMemHandle failed: %s\n", cudaGetErrorString(err));
        // Same-process IPC may be disallowed; fall through to peer test
    } else {
        printf("cudaIpcOpenMemHandle (same process) succeeded\n");
        printf("dev_a=%p dev_b=%p (same? %d)\n", dev_a, dev_b, dev_a == dev_b);

        // Write via dev_a, read via dev_b
        writer<<<1, 32>>>((int*)dev_a, 42);
        int* host_out;
        cudaMallocManaged(&host_out, sizeof(int));
        reader<<<1, 32>>>((int*)dev_b, host_out);
        cudaDeviceSynchronize();
        printf("Round-trip read via IPC handle: %d (expected 42)\n", *host_out);

        cudaIpcCloseMemHandle(dev_b);
    }

    // Time IPC handle export/import
    int N = 10000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaIpcGetMemHandle(&handle, dev_a);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double get_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    printf("cudaIpcGetMemHandle: %.2f us/call\n", get_us);

    cudaFree(dev_a);
    return 0;
}
