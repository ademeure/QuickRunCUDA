// cudaStreamGetCaptureInfo cost during/before capture
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 5; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ns = std::chrono::duration<float, std::nano>(t1-t0).count();
            if (ns < best) best = ns;
        }
        return best;
    };

    printf("# B300 cudaStreamGetCaptureInfo cost\n\n");

    // Outside capture
    {
        cudaStreamCaptureStatus status;
        unsigned long long id;
        float t = bench([&]{ cudaStreamGetCaptureInfo(s, &status, &id); });
        printf("  Outside capture: %.0f ns\n", t);
    }

    // During capture
    {
        cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
        cudaStreamCaptureStatus status;
        unsigned long long id;
        float t = bench([&]{ cudaStreamGetCaptureInfo(s, &status, &id); });
        cudaGraph_t g;
        cudaStreamEndCapture(s, &g);
        cudaGraphDestroy(g);
        printf("  During capture:  %.0f ns\n", t);
    }

    return 0;
}
