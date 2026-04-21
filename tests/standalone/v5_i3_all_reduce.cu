// V5 I3: Custom 2-GPU all-reduce SoL
// Pattern: each GPU has N floats, reduce-add to make both GPUs see global sum
// 2-GPU ring: each sends half to peer, peer adds, sends back
// Compare to peer-direct write + add
#include <cuda_runtime.h>
#include <cstdio>

__global__ void add_peer(float* local, const float* peer, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    for (int i = gtid; i < n; i += total) {
        local[i] += peer[i];
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    size_t n = 64 * 1024 * 1024;  // 64M floats = 256 MB
    size_t bytes = n * sizeof(float);

    cudaSetDevice(0);
    float* dev0;
    cudaMalloc(&dev0, bytes);
    cudaMemset(dev0, 0x3F, bytes);  // ~0.5

    cudaSetDevice(1);
    float* dev1;
    cudaMalloc(&dev1, bytes);
    cudaMemset(dev1, 0x3E, bytes);  // ~0.18

    cudaSetDevice(0);
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    int blocks = 296, threads = 256;

    // Warmup
    add_peer<<<blocks, threads>>>(dev0, dev1, n);
    cudaSetDevice(1);
    add_peer<<<blocks, threads>>>(dev1, dev0, n);
    cudaDeviceSynchronize();

    // 2-GPU all-reduce: each adds peer's data
    cudaSetDevice(0);
    cudaEventRecord(s);
    add_peer<<<blocks, threads>>>(dev0, dev1, n);   // GPU 0: dev0 += dev1 (read peer)
    cudaSetDevice(1);
    add_peer<<<blocks, threads>>>(dev1, dev0, n);   // GPU 1: dev1 += dev0 (read peer)
    cudaSetDevice(0);
    cudaDeviceSynchronize();
    cudaSetDevice(1);
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    cudaEventRecord(e);
    cudaEventSynchronize(e);
    float ms;
    cudaEventElapsedTime(&ms, s, e);

    double effective_bytes = 2 * bytes;  // each GPU reads peer's data
    printf("2-GPU all-reduce 256 MB:\n");
    printf("  time: %.3f ms\n", ms);
    printf("  effective BW (peer read): %.1f GB/s\n", effective_bytes / ms / 1e6);
    printf("  Compare NVLink peak: 956 GB/s; cudaMemcpyPeer: 749 GB/s (J2)\n");

    return 0;
}
