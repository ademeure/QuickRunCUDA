// V6 D3: Graph capture vs explicit construction
// METHOD A: cudaStreamBeginCapture + N kernel launches + cudaStreamEndCapture
// METHOD B: cudaGraphAddKernelNode × N (explicit)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf, int idx) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[idx] = idx;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4096);

    int N_NODES = 100;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Warmup
    noop<<<1, 32, 0, stream>>>(buf, 0);
    cudaStreamSynchronize(stream);

    // METHOD A: capture
    auto t0 = std::chrono::high_resolution_clock::now();
    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    for (int i = 0; i < N_NODES; i++) {
        noop<<<1, 32, 0, stream>>>(buf, i);
    }
    cudaGraph_t graph_capt;
    cudaStreamEndCapture(stream, &graph_capt);
    auto t1 = std::chrono::high_resolution_clock::now();
    double capt_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

    // METHOD B: explicit
    auto t2 = std::chrono::high_resolution_clock::now();
    cudaGraph_t graph_expl;
    cudaGraphCreate(&graph_expl, 0);
    cudaGraphNode_t prev_node = nullptr;
    for (int i = 0; i < N_NODES; i++) {
        cudaKernelNodeParams kparams = {};
        kparams.func = (void*)noop;
        kparams.gridDim = dim3(1, 1, 1);
        kparams.blockDim = dim3(32, 1, 1);
        void* kargs[2] = {&buf, &i};
        kparams.kernelParams = kargs;
        cudaGraphNode_t node;
        if (prev_node) {
            cudaGraphAddKernelNode(&node, graph_expl, &prev_node, 1, &kparams);
        } else {
            cudaGraphAddKernelNode(&node, graph_expl, nullptr, 0, &kparams);
        }
        prev_node = node;
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double expl_us = std::chrono::duration<double, std::micro>(t3 - t2).count();

    // Instantiate both, compare
    auto t4 = std::chrono::high_resolution_clock::now();
    cudaGraphExec_t exec_capt;
    cudaGraphInstantiate(&exec_capt, graph_capt, nullptr, nullptr, 0);
    auto t5 = std::chrono::high_resolution_clock::now();
    double inst_capt_us = std::chrono::duration<double, std::micro>(t5 - t4).count();

    auto t6 = std::chrono::high_resolution_clock::now();
    cudaGraphExec_t exec_expl;
    cudaGraphInstantiate(&exec_expl, graph_expl, nullptr, nullptr, 0);
    auto t7 = std::chrono::high_resolution_clock::now();
    double inst_expl_us = std::chrono::duration<double, std::micro>(t7 - t6).count();

    // Launch each
    int N_RUNS = 100;
    auto t8 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) cudaGraphLaunch(exec_capt, 0);
    cudaStreamSynchronize(0);
    auto t9 = std::chrono::high_resolution_clock::now();
    double launch_capt_us = std::chrono::duration<double, std::micro>(t9 - t8).count() / N_RUNS;

    auto t10 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) cudaGraphLaunch(exec_expl, 0);
    cudaStreamSynchronize(0);
    auto t11 = std::chrono::high_resolution_clock::now();
    double launch_expl_us = std::chrono::duration<double, std::micro>(t11 - t10).count() / N_RUNS;

    printf("Build %d-node graph:\n", N_NODES);
    printf("  Capture mode:    %.1f us (%.2f us/node)\n", capt_us, capt_us / N_NODES);
    printf("  Explicit add:    %.1f us (%.2f us/node)\n", expl_us, expl_us / N_NODES);
    printf("Instantiate:\n");
    printf("  Capture graph:   %.1f us\n", inst_capt_us);
    printf("  Explicit graph:  %.1f us\n", inst_expl_us);
    printf("Launch (avg of %d):\n", N_RUNS);
    printf("  Capture-built:   %.2f us/launch\n", launch_capt_us);
    printf("  Explicit-built:  %.2f us/launch\n", launch_expl_us);

    return 0;
}
