// V7 E6: Conditional graph SWITCH (CU_GRAPH_COND_TYPE_SWITCH)
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

__global__ void inc_kernel(unsigned int* counter, int idx) {
    if (threadIdx.x == 0) atomicAdd(counter + idx, 1);
}

int main() {
    cudaSetDevice(0);
    unsigned int* counter;
    cudaMallocManaged(&counter, 16);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphConditionalHandle handle;
    cudaGraphConditionalHandleCreate(&handle, graph, 0, 0);

    // SWITCH node with 4 cases
    cudaGraphNodeParams node_params = {};
    node_params.type = cudaGraphNodeTypeConditional;
    node_params.conditional.handle = handle;
    node_params.conditional.type = cudaGraphCondTypeSwitch;
    node_params.conditional.size = 4;  // 4 cases

    cudaGraphNode_t cond_node;
    cudaError_t err = cudaGraphAddNode(&cond_node, graph, nullptr, nullptr, 0, &node_params);
    if (err != cudaSuccess) {
        printf("SWITCH conditional: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Conditional SWITCH node added OK (4 cases)\n");

    // Add a kernel to each case body
    int idxs[4] = {0, 1, 2, 3};
    for (int i = 0; i < 4; i++) {
        cudaGraph_t body = node_params.conditional.phGraph_out[i];
        cudaGraphNode_t kn;
        cudaKernelNodeParams kp = {};
        kp.func = (void*)inc_kernel;
        kp.gridDim = dim3(1); kp.blockDim = dim3(32);
        void* args[2] = {&counter, &idxs[i]};
        kp.kernelParams = args;
        cudaGraphAddKernelNode(&kn, body, nullptr, 0, &kp);
    }

    cudaGraphExec_t exec;
    err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) {
        printf("Instantiate: %s\n", cudaGetErrorString(err));
        return 1;
    }
    printf("Instantiate OK\n");

    // Default cond=0 → case 0 should execute
    for (int i = 0; i < 4; i++) counter[i] = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaGraphLaunch(exec, 0);
    cudaStreamSynchronize(0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double launch_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("SWITCH graph launch (cond=0): %.2f us, counter=[%u,%u,%u,%u]\n",
           launch_us, counter[0], counter[1], counter[2], counter[3]);

    return 0;
}
