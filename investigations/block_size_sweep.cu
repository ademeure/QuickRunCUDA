// Block size sweep for compute-bound kernel
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void ffma_kernel(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    float b = threadIdx.x * 0.002f;
    float c = threadIdx.x * 0.003f;
    float d = threadIdx.x * 0.004f;
    for (int i = 0; i < iters; i++) {
        a = a*k1 + k2;
        b = b*k1 + k2;
        c = c*k1 + k2;
        d = d*k1 + k2;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x * blockDim.x + threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 148 * 1024 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int total_threads = 148 * 1024;  // Fixed total work

    printf("# B300 FFMA throughput vs block size (4-ILP, fixed total work)\n");
    printf("# Total threads: %d, FMAs each: 100k × 4 = 400k FFMA\n\n", total_threads);
    printf("# %-10s %-10s %-10s %-10s %-10s\n",
           "block", "blocks", "blk/SM", "ms", "TFLOPS");

    int iters = 100000;
    for (int block_size : {32, 64, 128, 256, 512, 1024}) {
        int blocks = total_threads / block_size;
        if (blocks > 65535) blocks = 65535;

        for (int i = 0; i < 3; i++) ffma_kernel<<<blocks, block_size>>>(d_out, iters, 1.0001f, 0.0001f);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            ffma_kernel<<<blocks, block_size>>>(d_out, iters, 1.0001f, 0.0001f);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }

        // Get occupancy
        int max_blocks_per_sm;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(&max_blocks_per_sm, ffma_kernel, block_size, 0);

        long total_flops = (long)blocks * block_size * iters * 4 * 2;
        double tflops = total_flops / (best/1000.0) / 1e12;
        printf("  %-10d %-10d %-10d %-10.3f %-10.1f\n",
               block_size, blocks, max_blocks_per_sm, best, tflops);
    }

    return 0;
}
