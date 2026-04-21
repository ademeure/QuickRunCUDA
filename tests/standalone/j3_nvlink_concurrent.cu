// J3: NVLink multiple in-flight ops scaling
// Issue 1, 2, 4, 8 concurrent cudaMemcpyAsync over different streams
// See if aggregate BW scales (parallel link usage) or saturates
#include <cuda_runtime.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cudaDeviceEnablePeerAccess(1, 0);
    cudaSetDevice(1);
    cudaDeviceEnablePeerAccess(0, 0);

    size_t total_bytes = 256ull * 1024 * 1024;  // 256 MB total

    cudaSetDevice(0);
    void* dst[16];
    cudaSetDevice(1);
    void* src[16];
    for (int i = 0; i < 16; i++) {
        cudaSetDevice(0);
        cudaMalloc(&dst[i], total_bytes / 16);
        cudaSetDevice(1);
        cudaMalloc(&src[i], total_bytes / 16);
    }

    cudaStream_t streams[16];
    cudaSetDevice(0);
    for (int i = 0; i < 16; i++) cudaStreamCreate(&streams[i]);

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    for (int N : {1, 2, 4, 8, 16}) {
        // Warmup
        for (int i = 0; i < N; i++) {
            cudaMemcpyPeerAsync(dst[i], 0, src[i], 1, total_bytes / 16, streams[i]);
        }
        cudaDeviceSynchronize();

        // Time N concurrent transfers
        cudaEventRecord(s);
        for (int i = 0; i < N; i++) {
            cudaMemcpyPeerAsync(dst[i], 0, src[i], 1, total_bytes / 16, streams[i]);
        }
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);
        size_t total_transferred = (size_t)N * (total_bytes / 16);
        printf("N=%2d  time=%.3f ms  total=%5zu MB  agg BW=%.1f GB/s  per-stream=%.1f GB/s\n",
               N, ms, total_transferred / (1024*1024),
               total_transferred / ms / 1e6,
               (total_bytes / 16) / ms / 1e6);
    }

    return 0;
}
