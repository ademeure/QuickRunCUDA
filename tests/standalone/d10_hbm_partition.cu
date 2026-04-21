// D10: L2 partitioning across HBM channels
// Test: vary STRIDE between thread reads at DRAM scale
// If certain strides bias toward a single L2 partition / HBM channel, BW drops
#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(128, 4) void read_strided(unsigned int* base, unsigned int stride_dwords, int iters, unsigned int* out) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    unsigned int sum = 0;
    for (int i = 0; i < iters; i++) {
        unsigned int idx = ((unsigned)i * total + (unsigned)gtid) * stride_dwords;
        idx &= ((1u << 25) - 1u);  // wrap to 128 MB
        sum ^= base[idx];
    }
    if (sum == 0xDEADBEEF) out[gtid] = sum;
}

int main() {
    cudaSetDevice(0);
    unsigned int* base;
    cudaMalloc(&base, 128ull * 1024 * 1024);  // 128 MB
    cudaMemset(base, 0x42, 128ull * 1024 * 1024);

    unsigned int* out;
    cudaMalloc(&out, 1024 * sizeof(unsigned int));

    int blocks = 296;
    int threads = 128;
    int iters = 100000;

    cudaEvent_t s, e;
    cudaEventCreate(&s);
    cudaEventCreate(&e);

    // Sweep strides
    for (unsigned int stride_dwords : {1u, 2u, 4u, 8u, 16u, 32u, 64u, 128u, 256u, 512u, 1024u, 2048u, 4096u, 8192u, 16384u, 32768u}) {
        // Warmup
        read_strided<<<blocks, threads>>>(base, stride_dwords, 1000, out);
        cudaDeviceSynchronize();

        cudaEventRecord(s);
        read_strided<<<blocks, threads>>>(base, stride_dwords, iters, out);
        cudaEventRecord(e);
        cudaEventSynchronize(e);
        float ms;
        cudaEventElapsedTime(&ms, s, e);

        size_t total_bytes = (size_t)blocks * threads * iters * 4;
        double bw_gbs = total_bytes / ms / 1e6;
        printf("stride=%5u dwords (%6u B): %.3f ms, %.1f GB/s effective\n",
               stride_dwords, stride_dwords * 4, ms, bw_gbs);
    }

    return 0;
}
