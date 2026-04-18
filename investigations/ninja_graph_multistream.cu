// Multi-stream graph launch — can we beat single-stream 512 ns/kernel?
#include <cuda_runtime.h>
#include <cstdio>

extern "C" __global__ void noop() {}

void bench(int n_streams, int N_per_graph) {
    cudaStream_t streams[8];
    cudaGraph_t graphs[8];
    cudaGraphExec_t execs[8];
    for (int s = 0; s < n_streams; s++) {
        cudaStreamCreate(&streams[s]);
        cudaStreamBeginCapture(streams[s], cudaStreamCaptureModeGlobal);
        for (int i = 0; i < N_per_graph; i++) noop<<<1, 32, 0, streams[s]>>>();
        cudaStreamEndCapture(streams[s], &graphs[s]);
        cudaGraphInstantiate(&execs[s], graphs[s], nullptr, nullptr, 0);
    }

    // Warmup
    for (int s = 0; s < n_streams; s++) cudaGraphLaunch(execs[s], streams[s]);
    cudaDeviceSynchronize();

    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    float best = 1e30f;
    for (int trial = 0; trial < 10; trial++) {
        cudaEventRecord(e0, streams[0]);
        for (int s = 0; s < n_streams; s++) cudaGraphLaunch(execs[s], streams[s]);
        cudaDeviceSynchronize();
        cudaEventRecord(e1, streams[0]);
        cudaEventSynchronize(e1);
        float ms; cudaEventElapsedTime(&ms, e0, e1);
        if (ms < best) best = ms;
    }
    long total_kernels = (long)n_streams * N_per_graph;
    double per_kernel_us = best * 1000.0 / total_kernels;
    printf("streams=%d N_per=%d total=%ld  %.3f ms = %.3f us/kernel\n",
           n_streams, N_per_graph, total_kernels, best, per_kernel_us);

    for (int s = 0; s < n_streams; s++) {
        cudaGraphExecDestroy(execs[s]);
        cudaGraphDestroy(graphs[s]);
        cudaStreamDestroy(streams[s]);
    }
}

int main() {
    cudaSetDevice(0);
    noop<<<1, 32>>>(); cudaDeviceSynchronize();
    printf("# Multi-stream graph launch sweep\n");
    bench(1, 10000);
    bench(2, 10000);
    bench(4, 10000);
    bench(8, 10000);
    return 0;
}
