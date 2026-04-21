// V6 D1: cudaGraphAddKernelNode latency
// Build a graph with N kernel nodes, measure per-node add cost.
// Compare to capture-mode and instantiate cost.
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf, int idx, int unused) {
    if (threadIdx.x == 0) buf[idx] = idx;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4096 * 4);

    int N_NODES = 100;
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // Method 1: cudaGraphAddKernelNode N times
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaGraphNode_t nodes[100];
    for (int i = 0; i < N_NODES; i++) {
        cudaKernelNodeParams params = {};
        params.func = (void*)noop;
        params.gridDim = dim3(1, 1, 1);
        params.blockDim = dim3(32, 1, 1);
        params.sharedMemBytes = 0;
        void* args[3] = {&buf, &i, &i};
        params.kernelParams = args;
        params.extra = nullptr;
        cudaGraphAddKernelNode(&nodes[i], graph, nullptr, 0, &params);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double add_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // Method 2: cudaGraphInstantiate
    auto t2 = std::chrono::high_resolution_clock::now();
    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    auto t3 = std::chrono::high_resolution_clock::now();
    double instantiate_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    // Method 3: cudaGraphLaunch (warm)
    cudaGraphLaunch(graph_exec, 0);
    cudaDeviceSynchronize();
    auto t4 = std::chrono::high_resolution_clock::now();
    cudaGraphLaunch(graph_exec, 0);
    cudaDeviceSynchronize();
    auto t5 = std::chrono::high_resolution_clock::now();
    double launch_us = std::chrono::duration<double, std::micro>(t5 - t4).count();

    printf("Graph build (cudaGraphAddKernelNode × %d): %.1f us = %.2f us/node\n",
           N_NODES, add_us, add_us / N_NODES);
    printf("Graph instantiate: %.1f us\n", instantiate_us);
    printf("Graph launch (warm, %d kernels): %.1f us = %.2f us/kernel\n",
           N_NODES, launch_us, launch_us / N_NODES);

    // Method 4: ExecUpdate vs Instantiate (test in-place graph update cost)
    cudaGraph_t graph2;
    cudaGraphCreate(&graph2, 0);
    cudaGraphNode_t nodes2[100];
    for (int i = 0; i < N_NODES; i++) {
        cudaKernelNodeParams params = {};
        params.func = (void*)noop;
        params.gridDim = dim3(1, 1, 1);
        params.blockDim = dim3(64, 1, 1);  // CHANGED block dim
        int p1 = i + 1000;
        void* args[3] = {&buf, &p1, &i};
        params.kernelParams = args;
        cudaGraphAddKernelNode(&nodes2[i], graph2, nullptr, 0, &params);
    }
    auto t6 = std::chrono::high_resolution_clock::now();
    cudaGraphExecUpdateResultInfo info;
    cudaGraphExecUpdate(graph_exec, graph2, &info);
    auto t7 = std::chrono::high_resolution_clock::now();
    double execupdate_us = std::chrono::duration<double, std::micro>(t7 - t6).count();
    printf("Graph ExecUpdate (replace 100 nodes): %.1f us = %.2f us/node\n",
           execupdate_us, execupdate_us / N_NODES);
    printf("ExecUpdate vs Instantiate ratio: %.1fx faster\n", instantiate_us / execupdate_us);

    return 0;
}
