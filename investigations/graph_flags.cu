#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    cudaGraph_t g;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeRelaxed);
    for (int i = 0; i < 100; i++) noop<<<1, 32, 0, s>>>();
    cudaStreamEndCapture(s, &g);

    auto bench_inst = [&](unsigned long long flags, const char *name) {
        cudaGraphExec_t exec;
        cudaError_t err;
        // Test once first to see if the flag works
        err = cudaGraphInstantiate(&exec, g, flags);
        if (err != cudaSuccess) {
            printf("  %-40s ERR: %s\n", name, cudaGetErrorString(err));
            return;
        }
        cudaGraphExecDestroy(exec);

        // Now bench
        float best_inst = 1e30f;
        for (int i = 0; i < 30; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaGraphInstantiate(&exec, g, flags);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best_inst) best_inst = us;
            cudaGraphExecDestroy(exec);
        }
        printf("  %-40s instant=%.1f us\n", name, best_inst);
    };

    printf("# B300 cudaGraphInstantiate flags (100-node graph)\n\n");

    bench_inst(0, "default (0)");
    bench_inst(cudaGraphInstantiateFlagAutoFreeOnLaunch, "AutoFreeOnLaunch");
    bench_inst(cudaGraphInstantiateFlagUseNodePriority, "UseNodePriority");
    bench_inst(cudaGraphInstantiateFlagUpload, "Upload");
    bench_inst(cudaGraphInstantiateFlagAutoFreeOnLaunch | cudaGraphInstantiateFlagUseNodePriority,
               "AutoFree + Priority");
    // Skip DeviceLaunch since it crashed

    cudaGraphDestroy(g);
    return 0;
}
