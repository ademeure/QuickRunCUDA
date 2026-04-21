// V7 E4: Graph clone + replay
// Test cudaGraphClone + multi-execute
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

    cudaGraph_t graph_orig;
    cudaGraphCreate(&graph_orig, 0);
    cudaGraphNode_t n;
    cudaKernelNodeParams kp = {};
    kp.func = (void*)noop;
    kp.gridDim = dim3(1); kp.blockDim = dim3(32);
    void* args[1] = {&buf};
    kp.kernelParams = args;
    cudaGraphAddKernelNode(&n, graph_orig, nullptr, 0, &kp);

    // Clone graph
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaGraph_t graph_clone;
    cudaError_t err = cudaGraphClone(&graph_clone, graph_orig);
    auto t1 = std::chrono::high_resolution_clock::now();
    double clone_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    if (err != cudaSuccess) {
        printf("Clone failed: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // Instantiate both
    cudaGraphExec_t exec_orig, exec_clone;
    cudaGraphInstantiate(&exec_orig, graph_orig, nullptr, nullptr, 0);
    cudaGraphInstantiate(&exec_clone, graph_clone, nullptr, nullptr, 0);

    // Launch both N times
    *buf = 0;
    int N = 100;
    for (int i = 0; i < N; i++) cudaGraphLaunch(exec_orig, 0);
    cudaStreamSynchronize(0);
    unsigned int orig_count = *buf;

    *buf = 0;
    for (int i = 0; i < N; i++) cudaGraphLaunch(exec_clone, 0);
    cudaStreamSynchronize(0);
    unsigned int clone_count = *buf;

    printf("cudaGraphClone: %.2f us\n", clone_us);
    printf("Original executed %d times: counter=%u (expect %d)\n", N, orig_count, N);
    printf("Clone    executed %d times: counter=%u (expect %d)\n", N, clone_count, N);

    return 0;
}
