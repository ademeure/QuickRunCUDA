// cudaMemcpy D2D peak vs kernel-based copy
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void __launch_bounds__(512, 4) kernel_copy(int4 *dst, const int4 *src, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < N - 7*stride; i += 8*stride) {
        int4 v0 = src[i];
        int4 v1 = src[i + stride];
        int4 v2 = src[i + 2*stride];
        int4 v3 = src[i + 3*stride];
        int4 v4 = src[i + 4*stride];
        int4 v5 = src[i + 5*stride];
        int4 v6 = src[i + 6*stride];
        int4 v7 = src[i + 7*stride];
        dst[i] = v0;
        dst[i + stride] = v1;
        dst[i + 2*stride] = v2;
        dst[i + 3*stride] = v3;
        dst[i + 4*stride] = v4;
        dst[i + 5*stride] = v5;
        dst[i + 6*stride] = v6;
        dst[i + 7*stride] = v7;
    }
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    size_t bytes = 2048ul * 1024 * 1024;  // 2 GB
    void *src; cudaMalloc(&src, bytes);
    void *dst; cudaMalloc(&dst, bytes);
    cudaMemset(src, 0xab, bytes);

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            cudaEventRecord(e0, s);
            fn();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 D2D copy methods (2 GB transfer)\n\n");

    // Method 1: cudaMemcpyAsync D2D
    {
        float t = bench([&]{
            cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToDevice, s);
        });
        // D2D = 1 read + 1 write (2x bytes touched)
        double bw = bytes / (t/1000.0) / 1e9;
        printf("  cudaMemcpyAsync D2D:    %.2f ms = %.0f GB/s (1 dir)\n", t, bw);
        printf("                                     = %.0f GB/s (counting both R+W)\n", bw * 2);
    }

    // Method 2: kernel copy
    {
        int N = bytes / 16;
        float t = bench([&]{
            kernel_copy<<<148, 512, 0, s>>>((int4*)dst, (int4*)src, N);
        });
        double bw = bytes / (t/1000.0) / 1e9;
        printf("  kernel int4 copy 8-ILP: %.2f ms = %.0f GB/s (1 dir)\n", t, bw);
        printf("                                     = %.0f GB/s (counting both R+W)\n", bw * 2);
    }

    return 0;
}
