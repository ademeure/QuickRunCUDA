// V8 M8: Driver thread affinity impact on multi-stream perf
// Run multi-stream launches with/without sched_setaffinity pinning
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <sched.h>
#include <pthread.h>
#include <unistd.h>

__global__ void noop(unsigned int* buf) {
    if (threadIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    cudaStream_t s[16];
    for (int i = 0; i < 16; i++) cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);

    // Warmup
    noop<<<1, 32, 0, s[0]>>>(buf);
    cudaDeviceSynchronize();

    int N_RUNS = 100;
    int N_STREAMS = 16;

    // Test 1: default CPU affinity
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        for (int k = 0; k < N_STREAMS; k++) noop<<<1, 32, 0, s[k]>>>(buf);
    }
    for (int k = 0; k < N_STREAMS; k++) cudaStreamSynchronize(s[k]);
    auto t1 = std::chrono::high_resolution_clock::now();
    double default_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // Test 2: pin to single CPU
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(0, &set);  // pin to CPU 0
    sched_setaffinity(0, sizeof(set), &set);

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        for (int k = 0; k < N_STREAMS; k++) noop<<<1, 32, 0, s[k]>>>(buf);
    }
    for (int k = 0; k < N_STREAMS; k++) cudaStreamSynchronize(s[k]);
    auto t3 = std::chrono::high_resolution_clock::now();
    double pinned_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;

    printf("Multi-stream (16 streams × %d runs):\n", N_RUNS);
    printf("  Default affinity:     %.2f us/iter\n", default_us);
    printf("  Pinned to CPU 0:      %.2f us/iter\n", pinned_us);
    printf("  Speedup: %.2fx\n", default_us / pinned_us);

    return 0;
}
