// V7 E5: Conditional graph WHILE loop (CU_GRAPH_COND_TYPE_WHILE)
// Build a graph that loops a kernel while a condition is met
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

__global__ void inc_kernel(unsigned int* counter) {
    if (threadIdx.x == 0) atomicAdd(counter, 1);
}

int main() {
    cudaSetDevice(0);
    unsigned int* counter;
    cudaMallocManaged(&counter, 4);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph, 0, 0);

    // WHILE node: body executes while condition is non-zero
    cudaGraphNodeParams node_params = {};
    node_params.type = cudaGraphNodeTypeConditional;
    node_params.conditional.handle = handle;
    node_params.conditional.type = cudaGraphCondTypeWhile;
    node_params.conditional.size = 1;

    cudaGraphNode_t cond_node;
    cudaError_t err = cudaGraphAddNode(&cond_node, graph, nullptr, nullptr, 0, &node_params);
    if (err != cudaSuccess) {
        printf("WHILE conditional: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Conditional WHILE node added OK\n");

    cudaGraph_t body = node_params.conditional.phGraph_out[0];

    // Body: increment counter
    cudaGraphNode_t body_kernel;
    cudaKernelNodeParams kp = {};
    kp.func = (void*)inc_kernel;
    kp.gridDim = dim3(1); kp.blockDim = dim3(32);
    void* args[1] = {&counter};
    kp.kernelParams = args;
    cudaGraphAddKernelNode(&body_kernel, body, nullptr, 0, &kp);

    cudaGraphExec_t exec;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        printf("Instantiate: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Instantiate OK\n");

    // With default cond=0, WHILE doesn't execute. Test 1 launch.
    *counter = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaGraphLaunch(exec, 0);
    cudaStreamSynchronize(0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double launch_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("WHILE graph launch (cond=0 default): %.2f us, counter=%u\n", launch_us, *counter);

    return 0;
}
