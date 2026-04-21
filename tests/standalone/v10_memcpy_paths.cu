// V10: memcpy paths — cudaMemcpyAsync vs kernel-based copy
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

// Kernel-based memcpy: float4 stream
__global__ void copy_kernel(float4* __restrict__ dst, const float4* __restrict__ src, size_t n) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (; i < n; i += stride) {
        dst[i] = src[i];
    }
}

int main() {
    cudaSetDevice(0);

    // Buffer sizes to test
    size_t sizes_mb[] = {1, 4, 16, 64, 256, 1024};
    int n_reps = 10;

    printf("=== memcpy paths: D2D bandwidth (10 reps avg) ===\n");
    printf("  size (MB) | cudaMemcpyAsync | kernel copy | speedup\n");

    for (size_t mb : sizes_mb) {
        size_t bytes = mb * 1024 * 1024;
        size_t n_fl4 = bytes / 16;

        float4* d_src;
        float4* d_dst;
        CK(cudaMalloc(&d_src, bytes));
        CK(cudaMalloc(&d_dst, bytes));

        cudaStream_t stream;
        cudaStreamCreate(&stream);

        // Warmup
        for (int i = 0; i < 3; i++) {
            cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, stream);
        }
        cudaStreamSynchronize(stream);

        // cudaMemcpyAsync
        cudaEvent_t e0, e1;
        cudaEventCreate(&e0); cudaEventCreate(&e1);
        cudaEventRecord(e0, stream);
        for (int i = 0; i < n_reps; i++) {
            cudaMemcpyAsync(d_dst, d_src, bytes, cudaMemcpyDeviceToDevice, stream);
        }
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms_api = 0;
        cudaEventElapsedTime(&ms_api, e0, e1);
        ms_api /= n_reps;
        double gbs_api = (bytes / 1e9) / (ms_api / 1000.0);

        // Kernel copy
        int blocks = 148 * 4;  // 4 blocks per SM
        int threads = 256;
        // Warmup
        for (int i = 0; i < 3; i++) copy_kernel<<<blocks, threads, 0, stream>>>(d_dst, d_src, n_fl4);
        cudaStreamSynchronize(stream);

        cudaEventRecord(e0, stream);
        for (int i = 0; i < n_reps; i++) {
            copy_kernel<<<blocks, threads, 0, stream>>>(d_dst, d_src, n_fl4);
        }
        cudaEventRecord(e1, stream);
        cudaEventSynchronize(e1);
        float ms_ker = 0;
        cudaEventElapsedTime(&ms_ker, e0, e1);
        ms_ker /= n_reps;
        double gbs_ker = (bytes / 1e9) / (ms_ker / 1000.0);

        // Both read+write, so BW includes both: × 2 for payload accounting
        printf("  %8zu | %6.1f GB/s (%5.2f ms) | %6.1f GB/s (%5.2f ms) | %.2fx\n",
            mb, gbs_api, ms_api, gbs_ker, ms_ker, gbs_ker / gbs_api);

        cudaFree(d_src); cudaFree(d_dst);
        cudaEventDestroy(e0); cudaEventDestroy(e1);
        cudaStreamDestroy(stream);
    }
    return 0;
}
