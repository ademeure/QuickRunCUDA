// CUDA Graph: capture / instantiate / launch / update costs
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}
extern "C" __global__ void with_args(float *out, int n, float v) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) out[tid] = v;
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    float *d_out; cudaMalloc(&d_out, 1024 * sizeof(float));

    auto bench = [&](auto fn, int trials = 100) {
        for (int i = 0; i < 5; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 CUDA Graph: capture / instantiate / launch / update costs\n\n");

    // Test 1: Capture + instantiate cost vs # nodes
    printf("## Test 1: Capture + instantiate vs # kernel nodes\n");
    printf("  %-12s %-15s %-15s %-15s %-15s\n",
           "n_nodes", "capture_us", "instant_us", "launch_us", "vs_streams");

    for (int n_nodes : {1, 10, 50, 200, 1000}) {
        cudaGraph_t g;
        cudaGraphExec_t exec;

        // Capture
        float t_cap = bench([&]{
            cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
            for (int i = 0; i < n_nodes; i++)
                noop<<<1, 32, 0, s>>>();
            cudaStreamEndCapture(s, &g);
            cudaGraphDestroy(g);
        }, 20);

        // Instantiate
        cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
        for (int i = 0; i < n_nodes; i++)
            noop<<<1, 32, 0, s>>>();
        cudaStreamEndCapture(s, &g);

        float t_inst = bench([&]{
            cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0);
            cudaGraphExecDestroy(exec);
        }, 20);

        cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0);

        // Launch
        float t_launch = bench([&]{
            cudaGraphLaunch(exec, s);
            cudaStreamSynchronize(s);
        }, 50);

        // Compare to streamed launches
        float t_stream = bench([&]{
            for (int i = 0; i < n_nodes; i++)
                noop<<<1, 32, 0, s>>>();
            cudaStreamSynchronize(s);
        }, 50);

        printf("  %-12d %-15.1f %-15.1f %-15.1f %-15.2fx\n",
               n_nodes, t_cap, t_inst, t_launch, t_stream / t_launch);

        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    }

    // Test 2: ExecUpdate vs reinstantiate
    printf("\n## Test 2: cudaGraphExecUpdate (change kernel args) vs reinstantiate\n");
    {
        int n_nodes = 100;
        cudaGraph_t g;
        cudaGraphExec_t exec;

        cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
        for (int i = 0; i < n_nodes; i++)
            with_args<<<1, 32, 0, s>>>(d_out, 32, 1.0f);
        cudaStreamEndCapture(s, &g);
        cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0);

        // Build a NEW graph with different args
        cudaGraph_t g_new;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
        for (int i = 0; i < n_nodes; i++)
            with_args<<<1, 32, 0, s>>>(d_out, 32, 2.0f);
        cudaStreamEndCapture(s, &g_new);

        // Method 1: ExecUpdate
        float t_update = bench([&]{
            cudaGraphExecUpdate(exec, g_new, nullptr);
        }, 50);

        // Method 2: destroy old, instantiate new
        cudaGraphExec_t e2;
        cudaGraphInstantiate(&e2, g_new, nullptr, nullptr, 0);
        float t_reinst = bench([&]{
            cudaGraphExecDestroy(e2);
            cudaGraphInstantiate(&e2, g_new, nullptr, nullptr, 0);
        }, 20);

        printf("  ExecUpdate (%d nodes):     %.1f us\n", n_nodes, t_update);
        printf("  Destroy+reinstantiate:     %.1f us\n", t_reinst);
        printf("  Update is %.1fx faster\n", t_reinst / t_update);

        cudaGraphExecDestroy(exec);
        cudaGraphExecDestroy(e2);
        cudaGraphDestroy(g);
        cudaGraphDestroy(g_new);
    }

    // Test 3: Single-node kernel update via cudaGraphExecKernelNodeSetParams
    printf("\n## Test 3: Single-node param update (no rebuild)\n");
    {
        cudaGraph_t g;
        cudaGraphExec_t exec;
        cudaGraphNode_t node;

        // Build manually
        cudaGraphCreate(&g, 0);
        cudaKernelNodeParams params = {};
        params.func = (void*)with_args;
        params.gridDim = dim3(1);
        params.blockDim = dim3(32);
        float val = 1.0f;
        int n = 32;
        void *args[] = {&d_out, &n, &val};
        params.kernelParams = args;
        cudaGraphAddKernelNode(&node, g, nullptr, 0, &params);
        cudaGraphInstantiate(&exec, g, nullptr, nullptr, 0);

        float v2 = 2.0f;
        void *args2[] = {&d_out, &n, &v2};
        cudaKernelNodeParams new_params = params;
        new_params.kernelParams = args2;

        float t_node_update = bench([&]{
            cudaGraphExecKernelNodeSetParams(exec, node, &new_params);
        }, 100);
        printf("  Per-node param update: %.2f us\n", t_node_update);

        cudaGraphExecDestroy(exec);
        cudaGraphDestroy(g);
    }

    return 0;
}
