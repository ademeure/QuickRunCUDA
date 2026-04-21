// V7 F1: Stream callback chain
// Test cudaStreamAddCallback (deprecated?) vs cudaLaunchHostFunc cascade
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <atomic>

std::atomic<int> counter1(0);
std::atomic<int> counter2(0);

void CUDART_CB cb1(void* data) {
    counter1.fetch_add(1, std::memory_order_relaxed);
}

void CUDART_CB cb2(void* data) {
    counter2.fetch_add(1, std::memory_order_relaxed);
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);

    int N = 1000;

    // Warmup
    cudaLaunchHostFunc(s, cb1, nullptr);
    cudaStreamSynchronize(s);

    // Test: chain of callbacks via cudaLaunchHostFunc
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaLaunchHostFunc(s, cb1, nullptr);
        cudaLaunchHostFunc(s, cb2, nullptr);
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double per_chain_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    printf("Callback CHAIN (cb1+cb2 per iter): %.2f us/pair\n", per_chain_us);
    printf("Counters: cb1=%d cb2=%d (expected %d each)\n", counter1.load(), counter2.load(), N + 1);

    return 0;
}
