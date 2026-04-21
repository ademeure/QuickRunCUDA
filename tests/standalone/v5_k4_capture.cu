// V5 K4: cudaStreamBeginCapture/EndCapture overhead
// Compare:
//   1. Capture N kernel launches into a graph
//   2. Direct N kernel launches
//   3. Explicit graph build (cudaGraphAddKernelNode)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);

    int N = 100;

    // Test 1: Direct launches (no graph)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) noop<<<1, 1, 0, s>>>();
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double direct_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("Direct %d launches: %.2f us total = %.2f us each\n", N, direct_us, direct_us / N);

    // Test 2: Capture + instantiate + launch
    auto t2 = std::chrono::high_resolution_clock::now();
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < N; i++) noop<<<1, 1, 0, s>>>();
    cudaGraph_t graph;
    cudaStreamEndCapture(s, &graph);
    auto t3 = std::chrono::high_resolution_clock::now();
    double capture_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    printf("Capture %d launches: %.2f us total\n", N, capture_us);

    cudaGraphExec_t exec;
    auto t4 = std::chrono::high_resolution_clock::now();
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    auto t5 = std::chrono::high_resolution_clock::now();
    double inst_us = std::chrono::duration<double, std::micro>(t5 - t4).count();
    printf("Instantiate: %.2f us\n", inst_us);

    auto t6 = std::chrono::high_resolution_clock::now();
    cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);
    auto t7 = std::chrono::high_resolution_clock::now();
    double launch_us = std::chrono::duration<double, std::micro>(t7 - t6).count();
    printf("Launch graph (%d nodes): %.2f us\n", N, launch_us);

    // Test 3: Re-launch (warm)
    auto t8 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {
        cudaGraphLaunch(exec, s);
    }
    cudaStreamSynchronize(s);
    auto t9 = std::chrono::high_resolution_clock::now();
    double relaunch_us = std::chrono::duration<double, std::micro>(t9 - t8).count() / 100;
    printf("Re-launch graph: %.2f us each\n", relaunch_us);

    return 0;
}
