// V6 D4: Multi-node graph dependency chain depth latency
// Compare:
//   N kernels in single linear chain (depth N)
//   N kernels in N independent chains (depth 1, parallel)
//   N kernels in sqrt(N) × sqrt(N) DAG
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[idx] = idx;
}

double bench_graph_topology(unsigned int* buf, int N, const char* name,
                            void (*build_fn)(cudaGraph_t, unsigned int*, int)) {
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);
    build_fn(graph, buf, N);

    cudaGraphExec_t exec;
    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);

    int N_RUNS = 100;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) cudaGraphLaunch(exec, 0);
    cudaStreamSynchronize(0);
    auto t1 = std::chrono::high_resolution_clock::now();
    double launch_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    return launch_us;
}

void build_linear_chain(cudaGraph_t graph, unsigned int* buf, int N) {
    cudaGraphNode_t prev = nullptr;
    for (int i = 0; i < N; i++) {
        cudaKernelNodeParams kp = {};
        kp.func = (void*)noop;
        kp.gridDim = dim3(1); kp.blockDim = dim3(32);
        void* args[2] = {&buf, &i};
        kp.kernelParams = args;
        cudaGraphNode_t node;
        if (prev) cudaGraphAddKernelNode(&node, graph, &prev, 1, &kp);
        else cudaGraphAddKernelNode(&node, graph, nullptr, 0, &kp);
        prev = node;
    }
}

void build_parallel(cudaGraph_t graph, unsigned int* buf, int N) {
    for (int i = 0; i < N; i++) {
        cudaKernelNodeParams kp = {};
        kp.func = (void*)noop;
        kp.gridDim = dim3(1); kp.blockDim = dim3(32);
        void* args[2] = {&buf, &i};
        kp.kernelParams = args;
        cudaGraphNode_t node;
        cudaGraphAddKernelNode(&node, graph, nullptr, 0, &kp);
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4096);

    // Warmup
    noop<<<1, 32>>>(buf, 0);
    cudaDeviceSynchronize();

    int N_VALS[] = {1, 4, 16, 64, 256};
    for (int idx = 0; idx < 5; idx++) {
        int N = N_VALS[idx];
        double linear_us = bench_graph_topology(buf, N, "linear", build_linear_chain);
        double parallel_us = bench_graph_topology(buf, N, "parallel", build_parallel);
        printf("N=%-3d  linear=%6.1f us (%.2f us/kernel)  parallel=%6.1f us (%.2f us/kernel)  ratio=%.2fx\n",
               N, linear_us, linear_us/N, parallel_us, parallel_us/N, linear_us/parallel_us);
    }

    return 0;
}
