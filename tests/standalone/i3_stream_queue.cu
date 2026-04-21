// I3: Host-side stream queue depth - how many kernels can be queued before host blocks?
// Setup: a kernel that takes ~10 ms (delays GPU). Then enqueue many launches into one stream.
// Time how long each enqueue takes. Find when host starts to block.
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void slow_kernel(int delay_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Spin for delay_iters cycles using clock64
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < (unsigned long long)delay_iters);
    }
}

int main() {
    cudaSetDevice(0);

    // Each kernel runs ~10 ms (= 15 M cy at 1500 MHz)
    int delay = 15000000;
    cudaStream_t s;
    cudaStreamCreate(&s);

    // Phase 1: enqueue many launches and measure host-side time per launch
    int N = 10000;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        slow_kernel<<<1, 32, 0, s>>>(delay);
        if (i == 99 || i == 999 || i == 9999) {
            auto t = std::chrono::high_resolution_clock::now();
            double elapsed_us = std::chrono::duration<double, std::micro>(t - t0).count();
            printf("After %5d enqueues: %.1f us total = %.2f us/enqueue\n",
                   i+1, elapsed_us, elapsed_us / (i+1));
        }
    }

    // Find approximate queue depth by timing first M enqueues
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);

    // Phase 2: enqueue increment until host blocks (>10 us per enqueue)
    cudaStreamCreate(&s);
    int prev_count = 0;
    for (int batch_size : {500, 600, 700, 800, 900, 1024}) {
        auto bt0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < batch_size; i++) {
            slow_kernel<<<1, 32, 0, s>>>(delay);
        }
        auto bt1 = std::chrono::high_resolution_clock::now();
        double batch_us = std::chrono::duration<double, std::micro>(bt1 - bt0).count();
        printf("Batch %4d: %.1f us total, %.2f us/enqueue\n",
               batch_size, batch_us, batch_us / batch_size);
        // wait so subsequent batches start fresh
        cudaStreamSynchronize(s);
    }

    return 0;
}
