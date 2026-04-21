// V10: cudaStreamSynchronize on empty stream — the "sync floor"
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);

    int N = 10000;

    // Warmup
    for (int i = 0; i < 100; i++) cudaStreamSynchronize(s);

    // Measure empty-stream sync
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaStreamSynchronize(s);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double us_empty = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // Measure: launch+sync vs just sync
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaStreamSynchronize(s);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double us_empty2 = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    // cudaEventRecord + Synchronize
    cudaEvent_t e;
    cudaEventCreate(&e);
    for (int i = 0; i < 100; i++) { cudaEventRecord(e, s); cudaEventSynchronize(e); }

    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaEventRecord(e, s);
        cudaEventSynchronize(e);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double us_event = std::chrono::duration<double, std::micro>(t5 - t4).count() / N;

    // cudaDeviceSynchronize overhead
    for (int i = 0; i < 100; i++) cudaDeviceSynchronize();
    auto t6 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaDeviceSynchronize();
    }
    auto t7 = std::chrono::high_resolution_clock::now();
    double us_dev = std::chrono::duration<double, std::micro>(t7 - t6).count() / N;

    printf("=== Sync overhead floor (host-side cost) ===\n");
    printf("  cudaStreamSynchronize empty:  %.3f us\n", us_empty);
    printf("  cudaStreamSynchronize empty2: %.3f us (sanity check)\n", us_empty2);
    printf("  cudaEventRecord + cudaEventSynchronize: %.3f us\n", us_event);
    printf("  cudaDeviceSynchronize empty: %.3f us\n", us_dev);

    cudaEventDestroy(e);
    cudaStreamDestroy(s);
    return 0;
}
