// Cost of stream capture in different modes
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 100) {
        for (int i = 0; i < 5; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 stream capture mode costs\n\n");

    // Test capture begin/end overhead
    printf("## Capture begin/end overhead (no kernels)\n");
    {
        cudaGraph_t g;
        float t = bench([&]{
            cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
            cudaStreamEndCapture(s, &g);
            cudaGraphDestroy(g);
        });
        printf("  Global mode:  %.2f us\n", t);
    }
    {
        cudaGraph_t g;
        float t = bench([&]{
            cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal);
            cudaStreamEndCapture(s, &g);
            cudaGraphDestroy(g);
        });
        printf("  ThreadLocal:  %.2f us\n", t);
    }
    {
        cudaGraph_t g;
        float t = bench([&]{
            cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
            cudaStreamEndCapture(s, &g);
            cudaGraphDestroy(g);
        });
        printf("  Relaxed:      %.2f us\n", t);
    }

    // Capture with N kernels
    printf("\n## Capture with N kernels\n");
    for (int N : {1, 10, 100, 1000}) {
        cudaGraph_t g;
        float t = bench([&]{
            cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
            for (int i = 0; i < N; i++) noop<<<1, 32, 0, s>>>();
            cudaStreamEndCapture(s, &g);
            cudaGraphDestroy(g);
        }, 10);
        printf("  %4d kernels: %.0f us = %.2f us per kernel\n", N, t, t/N);
    }

    return 0;
}
