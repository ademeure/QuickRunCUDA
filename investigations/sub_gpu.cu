// Test concurrent kernels using fraction of SMs each
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

extern "C" __global__ void busy(int *out, int iters) {
    float a = 1.0f + threadIdx.x * 0.001f;
    for (int i = 0; i < iters; i++)
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    if (threadIdx.x == 0) out[blockIdx.x] = (int)a;
}

int main() {
    cudaSetDevice(0);
    int *d_out; cudaMalloc(&d_out, 1024 * sizeof(int));

    auto bench = [&](int n_streams, int blocks_each, int trials=10) {
        std::vector<cudaStream_t> ss(n_streams);
        for (int i = 0; i < n_streams; i++) cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking);

        // Warmup
        for (int i = 0; i < n_streams; i++) busy<<<blocks_each, 128, 0, ss[i]>>>(d_out, 100000);
        cudaDeviceSynchronize();

        float best = 1e30f;
        for (int t = 0; t < trials; t++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n_streams; i++) busy<<<blocks_each, 128, 0, ss[i]>>>(d_out, 100000);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        for (auto &s : ss) cudaStreamDestroy(s);
        return best;
    };

    printf("# B300 concurrent kernels using fraction of SMs each\n");
    printf("# Each kernel = N blocks × 128 threads × 100k FFMA iters\n\n");
    printf("# %-10s %-12s %-15s %-15s\n",
           "n_streams", "blocks/k", "time_ms", "vs_sequential");

    int configs[][2] = {
        // {blocks per kernel, n streams}
        {148, 1},  // full GPU, 1 stream
        {74, 2},   // half GPU each, 2 streams (should fit on full GPU)
        {37, 4},   // quarter GPU, 4 streams
        {19, 8},   // eighth GPU, 8 streams
        {10, 16},  // 16 streams
        {5, 32},   // 32 streams
        {2, 64},   // 64 streams
        {1, 128},  // 128 streams
        {1, 148},  // 148 streams (one block per SM-equivalent count)
    };

    for (auto &c : configs) {
        int blocks_per = c[0];
        int n_streams = c[1];
        float t = bench(n_streams, blocks_per);
        // Compare to sequential time
        float seq = bench(1, blocks_per * n_streams) * 1.0f;  // same total work serial
        float speedup = seq / t;
        printf("  %-10d %-12d %-15.3f %.2fx (seq=%.3fms, total_blocks=%d)\n",
               n_streams, blocks_per, t, speedup, seq, blocks_per * n_streams);
    }

    cudaFree(d_out);
    return 0;
}
