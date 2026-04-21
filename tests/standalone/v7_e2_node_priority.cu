// V7 E2: Graph node priority
// Test cudaGraphKernelNodeSetAttribute with cudaKernelNodeAttributePriority
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0]++;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t n;
    cudaKernelNodeParams kp = {};
    kp.func = (void*)noop;
    kp.gridDim = dim3(1); kp.blockDim = dim3(32);
    void* args[1] = {&buf};
    kp.kernelParams = args;
    cudaGraphAddKernelNode(&n, graph, nullptr, 0, &kp);

    // Set priority attribute
    cudaKernelNodeAttrValue attr_val;
    attr_val.priority = -2;  // High priority (lower number = higher)
    cudaError_t err = cudaGraphKernelNodeSetAttribute(n, cudaLaunchAttributePriority, &attr_val);
    if (err != cudaSuccess) {
        printf("SetAttribute priority failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Get it back
    cudaKernelNodeAttrValue read_back;
    cudaGraphKernelNodeGetAttribute(n, cudaLaunchAttributePriority, &read_back);
    printf("Set priority=-2, read back=%d\n", read_back.priority);

    // Instantiate with FlagUseNodePriority
    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, cudaGraphInstantiateFlagUseNodePriority);

    *buf = 0;
    int N = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) cudaGraphLaunch(exec, 0);
    cudaStreamSynchronize(0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double launch_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    printf("Graph launch with priority node: %.2f us (counter=%u)\n", launch_us, *buf);

    return 0;
}
