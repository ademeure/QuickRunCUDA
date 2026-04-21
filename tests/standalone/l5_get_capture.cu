// L5: cudaStreamGetCaptureInfo cost
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);

    int N = 100000;

    // Test 1: idle stream
    cudaStreamCaptureStatus status;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaStreamGetCaptureInfo(s, &status, nullptr);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double idle_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
    printf("cudaStreamGetCaptureInfo (idle stream): %.1f ns/call\n", idle_ns);

    // Test 2: while capturing
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaStreamGetCaptureInfo(s, &status, nullptr);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double cap_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / N;
    printf("cudaStreamGetCaptureInfo (capturing): %.1f ns/call\n", cap_ns);
    cudaGraph_t graph;
    cudaStreamEndCapture(s, &graph);
    cudaGraphDestroy(graph);

    return 0;
}
