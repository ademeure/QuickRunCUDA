// V5 H1: cudaGraphExecKernelNodeSetParams perf — update kernel args in-place
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop_with_args(int a, int b, int c, int* out) {
    if (a + b + c == 12345 && threadIdx.x == 0) out[blockIdx.x] = a;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);
    int* dev_out;
    cudaMalloc(&dev_out, 1024);

    int N = 10000;

    // Build a graph with one kernel node
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaGraphNode_t node;
    cudaKernelNodeParams params = {};
    params.func = (void*)noop_with_args;
    params.gridDim = dim3(1);
    params.blockDim = dim3(1);
    params.sharedMemBytes = 0;
    int a = 1, b = 2, c = 3;
    void* args[] = {&a, &b, &c, &dev_out};
    params.kernelParams = args;
    cudaGraphAddKernelNode(&node, graph, nullptr, 0, &params);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    // Warmup
    cudaGraphLaunch(exec, s);
    cudaStreamSynchronize(s);

    // Test 1: cudaGraphLaunch only (no update)
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaGraphLaunch(exec, s);
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double launch_ns = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
    printf("cudaGraphLaunch: %.1f ns/launch\n", launch_ns);

    // Test 2: cudaGraphExecKernelNodeSetParams (update args) + launch
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        a = i;
        cudaGraphExecKernelNodeSetParams(exec, node, &params);
        cudaGraphLaunch(exec, s);
    }
    cudaStreamSynchronize(s);
    auto t3 = std::chrono::high_resolution_clock::now();
    double update_launch_ns = std::chrono::duration<double, std::nano>(t3 - t2).count() / N;
    printf("Update args + launch: %.1f ns each\n", update_launch_ns);
    printf("Update overhead: %.1f ns\n", update_launch_ns - launch_ns);

    // Test 3: full re-instantiate (worst case)
    cudaGraphExec_t exec2;
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; i++) {  // fewer iterations because slow
        cudaGraphInstantiate(&exec2, graph, nullptr, nullptr, 0);
        cudaGraphLaunch(exec2, s);
        cudaGraphExecDestroy(exec2);
    }
    cudaStreamSynchronize(s);
    auto t5 = std::chrono::high_resolution_clock::now();
    double instantiate_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / 100;
    printf("Re-instantiate + launch + destroy: %.2f us each\n", instantiate_us);

    return 0;
}
