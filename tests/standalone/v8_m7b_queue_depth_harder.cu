// V8 M7b: Probe kernel queue depth harder
// Use shorter delay to enqueue many more kernels; find first blocking launch
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void slow(unsigned int* buf, int delay_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do { asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)delay_iters);
        buf[0] = 1;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 4);

    cudaStream_t s;
    cudaStreamCreate(&s);

    // Launch 50K kernels with 1 ms each
    int delay = 1500000;  // 1 ms
    int N = 50000;

    // Warmup
    slow<<<1, 32, 0, s>>>(buf, delay);
    cudaStreamSynchronize(s);

    // Track per-launch times; find first "blocked" launch
    double* times = (double*)malloc(N * sizeof(double));

    auto t_all_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        slow<<<1, 32, 0, s>>>(buf, delay);
        auto t1 = std::chrono::high_resolution_clock::now();
        times[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    auto t_all_end = std::chrono::high_resolution_clock::now();

    double total_ms = std::chrono::duration<double, std::milli>(t_all_end - t_all_start).count();

    // Find first launch that took > 100 us (clear blocking signature)
    int first_block = -1;
    int first_severe_block = -1;  // > 1000 us
    double max_time = 0;
    int max_idx = -1;
    for (int i = 0; i < N; i++) {
        if (first_block == -1 && times[i] > 100) first_block = i;
        if (first_severe_block == -1 && times[i] > 1000) first_severe_block = i;
        if (times[i] > max_time) { max_time = times[i]; max_idx = i; }
    }

    printf("Launched %d kernels (1 ms each):\n", N);
    printf("  Host-enqueue wall time: %.2f ms\n", total_ms);
    printf("  Avg per-launch: %.3f us\n", total_ms * 1000 / N);
    printf("  First launch > 100 us: index %d\n", first_block);
    printf("  First launch > 1000 us: index %d\n", first_severe_block);
    printf("  Max launch: %.2f us at index %d\n", max_time, max_idx);
    printf("  Sample launch times (every 2000th):\n");
    for (int i = 0; i < N; i += 2000) printf("    [%5d] %.2f us\n", i, times[i]);

    // Sync at end
    cudaStreamSynchronize(s);
    free(times);
    return 0;
}
