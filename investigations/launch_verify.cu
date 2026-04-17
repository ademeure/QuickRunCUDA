// Independently verify Agent 17's launch latency findings
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);

    cudaStream_t s; cudaStreamCreate(&s);

    // Warmup
    for (int i = 0; i < 5; i++) {
        noop<<<1, 1, 0, s>>>();
        cudaDeviceSynchronize();
    }

    auto bench = [&](int blocks, int threads, int trials=20) {
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            cudaEventRecord(e0, s);
            noop<<<blocks, threads, 0, s>>>();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms;
            cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    int configs[][2] = {
        {1, 1},
        {1, 32},
        {32, 1},
        {148, 32},
        {148, 128},
        {148, 1024},
        {296, 1024},
        {1000, 32},
        {10000, 32},
        {100000, 32},
        {1000000, 32},
    };

    printf("# B300 launch latency verification\n");
    printf("# blocks × threads → measured time (min of 20 trials)\n\n");

    for (auto &cfg : configs) {
        float t = bench(cfg[0], cfg[1]);
        printf("  %7d × %5d  =  %10.3f µs (%.1f million blocks)\n",
               cfg[0], cfg[1], t * 1000, cfg[0] / 1e6);
    }

    return 0;
}
