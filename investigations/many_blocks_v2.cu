// Rigorous: measure FFMA throughput vs block count
// Theoretical: at full occ (8 blocks/SM × 148 = 1184), should be ~74 TFLOPS
// Hypothesis to test: does scheduler add overhead at 1M+ blocks?
#include <cuda_runtime.h>
#include <cstdio>

__launch_bounds__(256, 8) __global__ void ffma(float *out, int iters_per_block) {
    float a = threadIdx.x * 0.001f;
    float b = a + 0.001f;
    float c = b + 0.001f;
    float d = c + 0.001f;
    for (int i = 0; i < iters_per_block; i++) {
        a = a*1.0001f + 0.0001f;
        b = b*1.0001f + 0.0001f;
        c = c*1.0001f + 0.0001f;
        d = d*1.0001f + 0.0001f;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 256 * sizeof(float));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Hold TOTAL FMA work constant; vary block count
    long total_fmas_per_thread = (long)8 * 100000 * 8;  // 8 blk/SM × 100k iter × 8 SMs (8x base for plenty of work)
    // Each block has 256 threads, each does 4*iters FMAs

    auto bench = [&](int blocks, int iters_per) {
        for (int i = 0; i < 3; i++) ffma<<<blocks, 256>>>(d_out, iters_per);
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            cudaEventRecord(e0);
            ffma<<<blocks, 256>>>(d_out, iters_per);
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        long ops = (long)blocks * 256 * iters_per * 4 * 2;  // 4 chains × 2 FLOPs/FMA
        double tflops = ops / (best/1000.0) / 1e12;
        return std::pair<float, double>{best, tflops};
    };

    // PER-BLOCK runtime is critical — keep it ≥ 100 us so launch overhead is < 1%
    // At full occupancy (1184 blocks), 100k iter ≈ 50 us per block runtime (guess)
    // Total work scales with block × iters

    printf("# B300 FFMA TFLOPS vs block count - RIGOROUS\n");
    printf("# Theoretical peak at 2032 MHz: 76.96 TFLOPS\n");
    printf("# Total work held constant (where possible)\n\n");

    // Reference: 1184 blocks × 100k iter (we know hits ~74 TFLOPS)
    printf("# %-12s %-15s %-12s %-12s %-12s\n",
           "blocks", "iters/block", "per_blk_us", "time_ms", "TFLOPS");

    long total_iters = 1184L * 100000;  // ~constant ops

    for (int blocks : {296, 1184, 4736, 18944, 100000, 1000000}) {
        int iters_per = total_iters / blocks;
        if (iters_per < 100) iters_per = 100;
        auto [t, tflops] = bench(blocks, iters_per);
        // Estimated per-block runtime: kernel time / blocks-per-batch
        // batches = ceil(blocks / (148 SMs × 8 blocks/SM)) = ceil(blocks / 1184)
        float batches = (float)blocks / 1184.0f;
        float per_blk_us = (t * 1000) / batches;  // wall per batch / 1184 blocks
        printf("  %-12d %-15d %-12.1f %-12.3f %-12.1f\n",
               blocks, iters_per, per_blk_us, t, tflops);
    }

    return 0;
}
