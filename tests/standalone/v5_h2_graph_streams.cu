// V5 H2: cudaGraphLaunch across multiple streams
// Test: spread N graph launches across 1, 2, 4, 8, 16 streams
// Measure aggregate throughput
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop() {}

int main() {
    cudaSetDevice(0);

    // Build a simple graph
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t node;
    cudaKernelNodeParams params = {};
    params.func = (void*)noop;
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    void* args[] = {};
    params.kernelParams = args;
    cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    int N = 100000;

    // Test with various stream counts (round-robin)
    for (int n_streams : {1, 2, 4, 8, 16, 32}) {
        cudaStream_t streams[32];
        for (int i = 0; i < n_streams; i++) cudaStreamCreate(&streams[i]);

        // Warmup
        for (int i = 0; i < n_streams; i++) {
            cudaGraphLaunch(exec, streams[i]);
        }
        cudaDeviceSynchronize();

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            cudaGraphLaunch(exec, streams[i % n_streams]);
        }
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double ns_per = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
        printf("n_streams=%2d: %.1f ns/launch (%.2fM launches/sec)\n",
               n_streams, ns_per, 1000.0 / ns_per);

        for (int i = 0; i < n_streams; i++) cudaStreamDestroy(streams[i]);
    }

    return 0;
}
