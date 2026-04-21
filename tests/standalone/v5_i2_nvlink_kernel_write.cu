// V5 I2: NVLink streaming WRITE bandwidth at multi-block parallel
// Compare: kernel writes to peer GPU memory at varying block counts
#include <cuda_runtime.h>
#include <cstdio>

__global__ void write_peer(uint4* dst, int n_uint4_per_thread, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    uint4 v = make_uint4(gtid, gtid+1, gtid+2, u2);
    for (int i = 0; i < n_uint4_per_thread; i++) {
        dst[i * total + gtid] = v;
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    size_t bytes = 1024ull * 1024 * 1024;  // 1 GB transfer

    cudaSetDevice(1);
    uint4* peer_buf;
    cudaMalloc(&peer_buf, bytes);

    cudaSetDevice(0);
    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    for (int blocks : {32, 64, 148, 296, 592, 1184, 2368}) {
        int threads = 128;
        int total = blocks * threads;
        int n_uint4_per_thread = bytes / total / 16;
        size_t actual = (size_t)n_uint4_per_thread * total * 16;

        // Warmup
        write_peer<<<blocks, threads>>>(peer_buf, n_uint4_per_thread, 7);
        cudaDeviceSynchronize();

        cudaEventRecord(s);
        write_peer<<<blocks, threads>>>(peer_buf, n_uint4_per_thread, 7);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        printf("blocks=%5d threads=%d: %.3f ms = %.1f GB/s\n",
               blocks, threads, ms, actual / ms / 1e6);
    }

    return 0;
}
