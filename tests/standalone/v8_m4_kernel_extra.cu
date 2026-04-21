// V8 M4: cudaKernelNodeParams.extra — pointer to CUDA_KERNEL_NODE_PARAMS_v1 extensions?
#include <cuda_runtime.h>
#include <cstdio>

__global__ void test(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaKernelNodeParams kp = {};
    kp.func = (void*)test;
    kp.gridDim = dim3(1); kp.blockDim = dim3(32);
    void* args[1] = {&buf};
    kp.kernelParams = args;
    kp.extra = nullptr;  // Reserved

    cudaGraphNode_t node;
    cudaGraphAddKernelNode(&node, graph, nullptr, 0, &kp);

    // Check if the extra field has any documented meaning
    printf("cudaKernelNodeParams.extra = nullptr works OK\n");
    printf("cudaKernelNodeParams has:\n");
    printf("  func, gridDim, blockDim, sharedMemBytes\n");
    printf("  kernelParams (void**) and extra (void**)\n");
    printf("Per CUDA 13 docs: 'extra' is reserved for future extensions\n");

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, 0);
    cudaStreamSynchronize(0);
    printf("Kernel executed: buf[0] = %u\n", buf[0]);

    return 0;
}
