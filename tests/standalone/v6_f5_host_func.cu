// V6 F5: cudaLaunchHostFunc overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <atomic>

std::atomic<int> counter(0);

void CUDART_CB host_callback(void* userdata) {
    counter.fetch_add(1, std::memory_order_relaxed);
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s;
    cudaStreamCreate(&s);

    int N_RUNS = 1000;

    // Warmup
    cudaLaunchHostFunc(s, host_callback, nullptr);
    cudaStreamSynchronize(s);

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        cudaLaunchHostFunc(s, host_callback, nullptr);
    }
    cudaStreamSynchronize(s);
    auto t1 = std::chrono::high_resolution_clock::now();
    double per_call_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    printf("cudaLaunchHostFunc: %.2f us/call (counter=%d)\n", per_call_us, counter.load());

    return 0;
}
