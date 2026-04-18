// CUDA Graph minimum amortized launch latency
//
// Theoretical:
//   Catalog: 1 node = 2.15 us, 1000 nodes = 0.56 us/kernel
//   Question: what's the floor at 10K, 100K, 1M nodes?
//   Empty noop kernel runtime: ~3-4 ns (catalog)
//   So launch overhead alone could go arbitrarily low if amortized
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

void bench_graph(int N_nodes) {
    cudaStream_t s; cudaStreamCreate(&s);
    cudaGraph_t graph;
    cudaGraphExec_t exec;

    // Build graph: N noop kernels in a chain (force serial execution)
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < N_nodes; i++) {
        noop<<<1, 32, 0, s>>>();
    }
    cudaStreamEndCapture(s, &graph);
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    // Warmup
    for (int i = 0; i < 3; i++) cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    int n_iter = (N_nodes > 10000) ? 5 : (N_nodes > 1000 ? 20 : 50);
    float best_ms = 1e30f;
    for (int i = 0; i < n_iter; i++) {
        cudaEventRecord(e0, s);
        cudaGraphLaunch(exec, s);
        cudaEventRecord(e1, s);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best_ms) best_ms = ms;
    }
    double per_kernel_us = best_ms * 1000.0 / N_nodes;
    printf("Graph N=%-7d  total=%8.3f ms  per_kernel=%6.3f us\n", N_nodes, best_ms, per_kernel_us);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
}

int main() {
    cudaSetDevice(0);
    // Warmup
    noop<<<1, 32>>>();
    cudaDeviceSynchronize();

    bench_graph(1);
    bench_graph(10);
    bench_graph(100);
    bench_graph(1000);
    bench_graph(10000);
    bench_graph(100000);
    return 0;
}
