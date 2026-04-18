// C4: cudaGraph instantiation cost vs node count
// Measure cudaGraphInstantiate time as N grows
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void k_noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaGraph_t g;
    cudaGraphCreate(&g, 0);
    cudaGraphExec_t ge;

    // Warmup graph with 1 node
    {
        cudaGraph_t gw;
        cudaGraphCreate(&gw, 0);
        cudaKernelNodeParams kp = {};
        kp.func = (void*)k_noop;
        kp.gridDim = dim3(1); kp.blockDim = dim3(1);
        cudaGraphNode_t node;
        cudaGraphAddKernelNode(&node, gw, nullptr, 0, &kp);
        cudaGraphExec_t gew;
        cudaGraphInstantiate(&gew, gw, 0);
        cudaGraphLaunch(gew, s);
        cudaStreamSynchronize(s);
        cudaGraphExecDestroy(gew);
        cudaGraphDestroy(gw);
    }

    int sizes[] = {1, 4, 16, 64, 256, 1024, 4096, 16384};
    int n_sizes = sizeof(sizes)/sizeof(sizes[0]);
    printf("# cudaGraph instantiation cost vs node count (linear chain of noop kernels)\n");
    printf("# nodes   build_us   inst_us   launch_us   per_node_inst_us\n");
    for (int s_i = 0; s_i < n_sizes; s_i++) {
        int n = sizes[s_i];
        // Build graph
        auto t0 = std::chrono::high_resolution_clock::now();
        cudaGraph_t g;
        cudaGraphCreate(&g, 0);
        cudaGraphNode_t prev = nullptr;
        cudaKernelNodeParams kp = {};
        kp.func = (void*)k_noop;
        kp.gridDim = dim3(1); kp.blockDim = dim3(1);
        for (int i = 0; i < n; i++) {
            cudaGraphNode_t node;
            cudaGraphNode_t* deps = (prev ? &prev : nullptr);
            int n_deps = (prev ? 1 : 0);
            cudaGraphAddKernelNode(&node, g, deps, n_deps, &kp);
            prev = node;
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        double build_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        // Instantiate
        cudaGraphExec_t ge;
        auto t2 = std::chrono::high_resolution_clock::now();
        cudaGraphInstantiate(&ge, g, 0);
        auto t3 = std::chrono::high_resolution_clock::now();
        double inst_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

        // Launch
        auto t4 = std::chrono::high_resolution_clock::now();
        cudaGraphLaunch(ge, s);
        cudaStreamSynchronize(s);
        auto t5 = std::chrono::high_resolution_clock::now();
        double launch_us = std::chrono::duration<double, std::micro>(t5 - t4).count();

        printf("  %5d  %9.1f  %8.1f  %9.1f  %.3f\n",
               n, build_us, inst_us, launch_us, inst_us / n);

        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }
    return 0;
}
