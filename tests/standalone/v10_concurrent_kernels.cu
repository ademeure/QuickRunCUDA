// V10: concurrent kernel limit — how many truly run in parallel?
// Launch N independent kernels (1 block each) on N streams
// If all run in parallel → time ≈ single kernel time
// If sequential (no concurrency) → time ≈ N × single kernel
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

__global__ void spin_kernel(int iters, int* sink) {
    int v = iters;
    #pragma unroll 1
    for (int i = 0; i < iters; i++) {
        v = v * 1664525 + 1013904223;
    }
    if (v == 0xdeadbeef) *sink = v;  // Real side effect prevents DCE
}

int main() {
    cudaSetDevice(0);

    int* d_sink;
    cudaMalloc(&d_sink, sizeof(int));
    int N_STREAMS_MAX = 256;
    cudaStream_t* streams = new cudaStream_t[N_STREAMS_MAX];
    for (int i = 0; i < N_STREAMS_MAX; i++) {
        cudaStreamCreateWithFlags(&streams[i], cudaStreamNonBlocking);
    }

    // Find iter count giving ~1 ms per kernel (solo)
    int iters = 500000;  // tune — want 0.5-2 ms per kernel
    // Warmup
    for (int i = 0; i < 10; i++) spin_kernel<<<1, 32, 0, streams[0]>>>(iters, d_sink);
    cudaStreamSynchronize(streams[0]);

    // Measure solo time
    auto t0 = std::chrono::high_resolution_clock::now();
    spin_kernel<<<1, 32, 0, streams[0]>>>(iters, d_sink);
    cudaStreamSynchronize(streams[0]);
    auto t1 = std::chrono::high_resolution_clock::now();
    double solo_us = std::chrono::duration<double, std::micro>(t1 - t0).count();
    printf("Solo kernel: %.1f us\n\n", solo_us);

    // Sweep N = 1, 2, 4, 8, ..., 256
    printf("=== Concurrent kernel scaling ===\n");
    printf("N streams  | total time | per-kernel | speedup vs sequential\n");
    for (int N : {1, 2, 4, 8, 16, 32, 64, 128, 148, 160, 200, 256}) {
        if (N > N_STREAMS_MAX) continue;
        // Launch N kernels on N streams
        for (int w = 0; w < 3; w++) {
            for (int i = 0; i < N; i++) spin_kernel<<<1, 32, 0, streams[i]>>>(iters, d_sink);
            for (int i = 0; i < N; i++) cudaStreamSynchronize(streams[i]);
        }

        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) spin_kernel<<<1, 32, 0, streams[i]>>>(iters, d_sink);
        for (int i = 0; i < N; i++) cudaStreamSynchronize(streams[i]);
        auto t3 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration<double, std::micro>(t3 - t2).count();
        double per_kernel = us / N;
        double speedup = (solo_us * N) / us;
        printf("  %4d     | %8.1f us | %7.2f us | %.2fx\n",
               N, us, per_kernel, speedup);
    }

    for (int i = 0; i < N_STREAMS_MAX; i++) cudaStreamDestroy(streams[i]);
    delete[] streams;
    return 0;
}
