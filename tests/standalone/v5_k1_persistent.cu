// V5 K1: Persistent kernel SoL — minimum overhead per task
// CPU writes a flag → persistent GPU kernel detects it, runs work, signals done
// Compare to launching a new kernel per task
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void persistent(volatile unsigned int* in_flag,
                           volatile unsigned int* out_flag,
                           int n_tasks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int t = 1; t <= n_tasks; t++) {
        // Wait for CPU to set in_flag = t
        while (*in_flag != (unsigned)t) { /* spin */ }
        // "Do work" — minimal (just signal done)
        *out_flag = (unsigned)t;
    }
}

__global__ void per_task(volatile unsigned int* out_flag, unsigned int task_id) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *out_flag = task_id;
}

int main() {
    cudaSetDevice(0);
    volatile unsigned int* in_flag;
    volatile unsigned int* out_flag;
    cudaMallocManaged((void**)&in_flag, sizeof(unsigned int));
    cudaMallocManaged((void**)&out_flag, sizeof(unsigned int));
    *in_flag = 0;
    *out_flag = 0;

    cudaStream_t s;
    cudaStreamCreate(&s);

    int N = 1000;

    // Test 1: persistent kernel (1 launch, N tasks via flag updates)
    persistent<<<1, 32, 0, s>>>(in_flag, out_flag, N);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int t = 1; t <= N; t++) {
        *in_flag = (unsigned)t;
        while (*out_flag != (unsigned)t) { /* spin CPU */ }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaStreamSynchronize(s);
    double persistent_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    *in_flag = 0; *out_flag = 0;
    cudaDeviceSynchronize();

    // Test 2: per-task kernel launch
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int t = 1; t <= N; t++) {
        per_task<<<1, 1, 0, s>>>(out_flag, (unsigned)t);
        while (*out_flag != (unsigned)t) { /* spin CPU */ }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double launch_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    printf("Persistent kernel (1 launch, N tasks): %.2f us per task\n", persistent_us);
    printf("Per-task kernel launch (N launches): %.2f us per task\n", launch_us);
    printf("Persistent speedup: %.2fx\n", launch_us / persistent_us);

    return 0;
}
