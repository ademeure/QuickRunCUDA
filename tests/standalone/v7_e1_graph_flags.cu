// V7 E1: cudaGraphInstantiate flags behavior
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    cudaGraphNode_t node;
    cudaKernelNodeParams kp = {};
    kp.func = (void*)noop;
    kp.gridDim = dim3(1); kp.blockDim = dim3(32);
    void* args[1] = {&buf};
    kp.kernelParams = args;
    cudaGraphAddKernelNode(&node, graph, nullptr, 0, &kp);

    // Flag variants to test:
    // 0 — default
    // cudaGraphInstantiateFlagAutoFreeOnLaunch (1)
    // cudaGraphInstantiateFlagUpload (2)
    // cudaGraphInstantiateFlagDeviceLaunch (4)
    // cudaGraphInstantiateFlagUseNodePriority (8)

    unsigned int flags[] = {0, 1, 2, 4, 8};
    const char* names[] = {"default", "AutoFreeOnLaunch", "Upload", "DeviceLaunch", "UseNodePriority"};

    for (int i = 0; i < 5; i++) {
        cudaGraphExec_t exec;
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaError_t err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, flags[i]);
        auto t1 = std::chrono::high_resolution_clock::now();
        double inst_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        if (err != cudaSuccess) {
            printf("Flag=%-18s %s\n", names[i], cudaGetErrorString(err));
            continue;
        }

        // Measure launch speed with this flag
        int N = 500;
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int j = 0; j < N; j++) cudaGraphLaunch(exec, 0);
        cudaStreamSynchronize(0);
        auto t3 = std::chrono::high_resolution_clock::now();
        double launch_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

        printf("Flag=%-18s Inst=%6.1f us  Launch=%5.2f us\n", names[i], inst_us, launch_us);
        cudaGraphExecDestroy(exec);
    }

    return 0;
}
