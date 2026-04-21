// V6 D5: Conditional CUDA Graph nodes (CUDA 12.3+ / 13)
// Test: build a graph with an IF node, instantiate, launch with cond=1 vs cond=0
// Compare to unconditional graph baseline
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

__global__ void noop_kernel(unsigned int* buf, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[idx] = idx + 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 1024);

    // Build a graph with one IF node
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    // Create conditional handle
    cudaGraphConditionalHandle cond_handle;
    cudaGraphConditionalHandleCreate(&cond_handle, graph, 0, 0);

    // Add conditional IF node
    cudaGraphNodeParams node_params = {};
    node_params.type = cudaGraphNodeTypeConditional;
    node_params.conditional.handle = cond_handle;
    node_params.conditional.type = cudaGraphCondTypeIf;
    node_params.conditional.size = 1;  // 1 body graph
    cudaGraphNode_t cond_node;
    cudaError_t err = cudaGraphAddNode(&cond_node, graph, nullptr, nullptr, 0, &node_params);
    if (err != cudaSuccess) {
        printf("cudaGraphAddNode (conditional) ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Conditional IF node added OK\n");

    // Get the body graph and add a kernel to it
    cudaGraph_t body_graph = node_params.conditional.phGraph_out[0];

    cudaGraphNode_t kernel_node;
    cudaKernelNodeParams kparams = {};
    kparams.func = (void*)noop_kernel;
    kparams.gridDim = dim3(1, 1, 1);
    kparams.blockDim = dim3(32, 1, 1);
    int idx_val = 7;
    void* kargs[2] = {&buf, &idx_val};
    kparams.kernelParams = kargs;
    cudaGraphAddKernelNode(&kernel_node, body_graph, nullptr, 0, &kparams);

    // Instantiate
    cudaGraphExec_t graph_exec;
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    buf[7] = 0;
    // Default launch value was 0, so condition is false → kernel skipped.
    // To set condition, must use device-side cudaGraphSetConditional or recreate handle.
    int N_RUNS = 1000;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        cudaGraphLaunch(graph_exec, 0);
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    double cond_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    printf("Conditional graph launch (cond=0 default): %.2f us/launch  buf[7]=%u\n", cond_us, buf[7]);

    // Compare to unconditional baseline
    cudaGraph_t baseline_graph;
    cudaGraphCreate(&baseline_graph, 0);
    cudaGraphNode_t base_node;
    cudaGraphAddKernelNode(&base_node, baseline_graph, nullptr, 0, &kparams);
    cudaGraphExec_t baseline_exec;
    cudaGraphInstantiate(&baseline_exec, baseline_graph, nullptr, nullptr, 0);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        cudaGraphLaunch(baseline_exec, 0);
    }
    cudaDeviceSynchronize();
    auto t3 = std::chrono::high_resolution_clock::now();
    double base_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    printf("Unconditional graph launch:                %.2f us/launch  buf[7]=%u\n", base_us, buf[7]);
    printf("Conditional overhead: %.2f us (%.2fx)\n", cond_us - base_us, cond_us / base_us);

    return 0;
}
