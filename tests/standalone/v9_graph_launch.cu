// V9: cudaGraph launch latency vs direct launch
// Measures host-side overhead per launch (kernel = empty)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e));exit(1);} }while(0)

__global__ void empty_kernel() {}

int main() {
    cudaSetDevice(0);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    int N = 1000;

    // Warm up
    for (int i = 0; i < 10; i++) empty_kernel<<<1, 32, 0, stream>>>();
    cudaStreamSynchronize(stream);

    // === DIRECT LAUNCH ===
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        empty_kernel<<<1, 32, 0, stream>>>();
    }
    cudaStreamSynchronize(stream);
    auto t1 = std::chrono::high_resolution_clock::now();
    double direct_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // === GRAPH LAUNCH (capture once, replay N times) ===
    cudaGraph_t graph;
    cudaGraphExec_t graphExec;

    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    empty_kernel<<<1, 32, 0, stream>>>();
    CK(cudaStreamEndCapture(stream, &graph));
    CK(cudaGraphInstantiate(&graphExec, graph, NULL, NULL, 0));

    // Warmup graph
    for (int i = 0; i < 10; i++) cudaGraphLaunch(graphExec, stream);
    cudaStreamSynchronize(stream);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaGraphLaunch(graphExec, stream);
    }
    cudaStreamSynchronize(stream);
    auto t3 = std::chrono::high_resolution_clock::now();
    double graph_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    // === PRE-RECORDED GRAPH (multiple kernels in one graph) ===
    cudaGraph_t graph2;
    cudaGraphExec_t graphExec2;
    CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i = 0; i < 100; i++) empty_kernel<<<1, 32, 0, stream>>>();
    CK(cudaStreamEndCapture(stream, &graph2));
    CK(cudaGraphInstantiate(&graphExec2, graph2, NULL, NULL, 0));

    for (int i = 0; i < 10; i++) cudaGraphLaunch(graphExec2, stream);
    cudaStreamSynchronize(stream);

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N/100; i++) {
        cudaGraphLaunch(graphExec2, stream);
    }
    cudaStreamSynchronize(stream);
    auto t5 = std::chrono::high_resolution_clock::now();
    double batched_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N;

    printf("=== Empty kernel launch latency (single thread, single block, host-side avg per launch) ===\n");
    printf("  Direct cudaLaunchKernel: %.2f us/launch\n", direct_us);
    printf("  Graph (1-kernel graph):  %.2f us/launch\n", graph_us);
    printf("  Graph (100-kernel batch): %.2f us/launch (amortized)\n", batched_us);
    printf("  Graph speedup vs direct: %.2fx (single), %.2fx (batched)\n",
           direct_us / graph_us, direct_us / batched_us);

    cudaGraphExecDestroy(graphExec);
    cudaGraphExecDestroy(graphExec2);
    cudaGraphDestroy(graph);
    cudaGraphDestroy(graph2);
    cudaStreamDestroy(stream);
    return 0;
}
