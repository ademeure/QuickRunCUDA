// V5 I5: UVA peer access — read peer GPU memory WITHOUT explicit P2P enable
// Test: does it work at all? what's the perf?
#include <cuda_runtime.h>
#include <cstdio>

__global__ void read_peer(uint4* src, uint4* sink, int n) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    uint4 acc = make_uint4(0,0,0,0);
    for (int i = 0; i < n; i++) {
        uint4 v = src[i * total + gtid];
        acc.x ^= v.x; acc.y ^= v.y; acc.z ^= v.z; acc.w ^= v.w;
    }
    if (acc.x == 0xDEADBEEF) sink[gtid] = acc;
}

int main() {
    int can_p2p;
    cudaDeviceCanAccessPeer(&can_p2p, 0, 1);
    printf("Can device 0 access device 1? %s\n", can_p2p ? "YES" : "NO");

    size_t bytes = 256ull * 1024 * 1024;  // 256 MB
    int blocks = 296, threads = 128;
    int total = blocks * threads;
    int n_uint4_per_thread = bytes / total / 16;
    size_t actual = (size_t)n_uint4_per_thread * total * 16;

    cudaSetDevice(1);
    uint4* peer_buf;
    cudaMalloc(&peer_buf, actual);
    cudaMemset(peer_buf, 0x42, actual);

    cudaSetDevice(0);
    uint4* sink;
    cudaMalloc(&sink, sizeof(uint4) * total);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // Test 1: WITHOUT enabling P2P (UVA only)
    printf("\n=== UVA only (no peer access enabled) ===\n");
    cudaError_t err1;
    read_peer<<<blocks, threads>>>(peer_buf, sink, n_uint4_per_thread);
    err1 = cudaDeviceSynchronize();
    if (err1 != cudaSuccess) {
        printf("UVA-only access FAILED: %s\n", cudaGetErrorString(err1));
    } else {
        cudaEventRecord(s);
        read_peer<<<blocks, threads>>>(peer_buf, sink, n_uint4_per_thread);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        printf("UVA peer read: %.3f ms = %.1f GB/s\n", ms, actual / ms / 1e6);
    }

    // Test 2: WITH explicit P2P
    printf("\n=== With cudaDeviceEnablePeerAccess(1, 0) ===\n");
    cudaSetDevice(0);
    err1 = cudaDeviceEnablePeerAccess(1, 0);
    if (err1 != cudaSuccess && err1 != cudaErrorPeerAccessAlreadyEnabled) {
        printf("EnablePeerAccess failed: %s\n", cudaGetErrorString(err1));
    } else {
        read_peer<<<blocks, threads>>>(peer_buf, sink, n_uint4_per_thread);
        cudaDeviceSynchronize();
        cudaEventRecord(s);
        read_peer<<<blocks, threads>>>(peer_buf, sink, n_uint4_per_thread);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        printf("P2P peer read: %.3f ms = %.1f GB/s\n", ms, actual / ms / 1e6);
    }

    return 0;
}
