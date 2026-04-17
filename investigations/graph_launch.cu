// CUDA Graph launch overhead vs direct launch — detailed
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop_kernel(int *out, int val) {
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = val;
}

int main() {
    cudaSetDevice(0);
    int *d_out;
    cudaMalloc(&d_out, sizeof(int));
    cudaStream_t s; cudaStreamCreate(&s);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);

    auto bench = [&](auto fn, int trials=20) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            cudaEventRecord(t0, s);
            fn();
            cudaEventRecord(t1, s);
            cudaEventSynchronize(t1);
            float ms; cudaEventElapsedTime(&ms, t0, t1);
            if (ms < best) best = ms;
        }
        return best;
    };

    // Direct launches (N kernels)
    for (int N : {1, 8, 32, 128, 1024}) {
        float t_direct = bench([&]{
            for (int i = 0; i < N; i++) {
                noop_kernel<<<1, 1, 0, s>>>(d_out, i);
            }
        });

        // Capture to graph
        cudaGraph_t g;
        cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);
        for (int i = 0; i < N; i++) {
            noop_kernel<<<1, 1, 0, s>>>(d_out, i);
        }
        cudaStreamEndCapture(s, &g);

        cudaGraphExec_t ge;
        cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0);

        float t_graph = bench([&]{
            cudaGraphLaunch(ge, s);
        });

        printf("  N=%-5d : direct=%.3f us, graph=%.3f us (%.2f× faster)\n",
               N, t_direct*1000, t_graph*1000, t_direct/t_graph);

        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
    }

    cudaFree(d_out);
    return 0;
}
