// V6 D2: ExecUpdate vs Instantiate — detailed comparison at various graph sizes
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[idx] = idx;
}

void build_graph(cudaGraph_t graph, unsigned int* buf, int N, int offset) {
    cudaGraphNode_t prev = nullptr;
    for (int i = 0; i < N; i++) {
        cudaKernelNodeParams kp = {};
        kp.func = (void*)noop;
        kp.gridDim = dim3(1); kp.blockDim = dim3(32);
        int idx_val = i + offset;
        void* args[2] = {&buf, &idx_val};
        kp.kernelParams = args;
        cudaGraphNode_t node;
        if (prev) cudaGraphAddKernelNode(&node, graph, &prev, 1, &kp);
        else cudaGraphAddKernelNode(&node, graph, nullptr, 0, &kp);
        prev = node;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4096);
    noop<<<1, 32>>>(buf, 0);
    cudaDeviceSynchronize();

    int sizes[] = {10, 50, 100, 500, 1000};
    printf("Graph size vs Instantiate vs ExecUpdate:\n");
    printf("%-6s %-15s %-15s %-10s\n", "N", "Instantiate(us)", "ExecUpdate(us)", "Speedup");
    printf("---------------------------------------------------------\n");

    for (int idx = 0; idx < 5; idx++) {
        int N = sizes[idx];

        cudaGraph_t g1, g2;
        cudaGraphCreate(&g1, 0);
        build_graph(g1, buf, N, 0);

        // Time instantiate
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaGraphExec_t exec;
        cudaGraphInstantiate(&exec, g1, nullptr, nullptr, 0);
        auto t1 = std::chrono::high_resolution_clock::now();
        double inst_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        // Build new graph with same topology, different params
        cudaGraphCreate(&g2, 0);
        build_graph(g2, buf, N, 1000);

        // Time ExecUpdate
        auto t2 = std::chrono::high_resolution_clock::now();
        cudaGraphExecUpdateResultInfo info;
        cudaError_t err = cudaGraphExecUpdate(exec, g2, &info);
        auto t3 = std::chrono::high_resolution_clock::now();
        double upd_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

        if (err != cudaSuccess) {
            printf("%-6d %-15.1f UPDATE FAILED: %s\n", N, inst_us, cudaGetErrorString(err));
        } else {
            printf("%-6d %-15.1f %-15.1f %.1fx\n", N, inst_us, upd_us, inst_us / upd_us);
        }

        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g1);
        cudaGraphDestroy(g2);
    }

    return 0;
}
