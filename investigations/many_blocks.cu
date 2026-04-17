// Effect of huge block count on FFMA throughput
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void ffma(float *out, int iters, float k1, float k2) {
    float a = threadIdx.x * 0.001f;
    float b = a + 0.001f;
    float c = b + 0.001f;
    float d = c + 0.001f;
    for (int i = 0; i < iters; i++) {
        a = a*k1 + k2; b = b*k1 + k2; c = c*k1 + k2; d = d*k1 + k2;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 1024 * 256 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    int iters_total = 148 * 1024 * 100000;  // total work amount = 148 SMs × 1024 thr × 100k iter

    auto bench = [&](int blocks, int iters_per_block) {
        for (int i = 0; i < 3; i++) ffma<<<blocks, 256>>>(d_out, iters_per_block, 1.0001f, 0.0001f);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            ffma<<<blocks, 256>>>(d_out, iters_per_block, 1.0001f, 0.0001f);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * 256 * iters_per_block * 4 * 2;
        double tflops = ops / (best/1000.0) / 1e12;
        return std::pair<float, double>{best, tflops};
    };

    printf("# B300 FFMA throughput vs block count (constant total work)\n");
    printf("# Each kernel does ~10 ms of FFMA work\n\n");
    printf("# %-12s %-15s %-12s %-12s\n", "blocks", "iters/block", "time_ms", "TFLOPS");

    // Always 256 threads/block, vary blocks and iters to keep total work constant
    int total_work = 1184 * 100000;  // base work for 1184 blocks (8 blk/SM × 148)
    for (int blocks : {148, 296, 592, 1184, 2368, 4736, 9472, 18944, 37888, 100000, 1000000}) {
        int iters_per = total_work / blocks;
        if (iters_per < 100) iters_per = 100;
        auto [t, tflops] = bench(blocks, iters_per);
        printf("  %-12d %-15d %-12.3f %-12.1f\n", blocks, iters_per, t, tflops);
    }

    return 0;
}
