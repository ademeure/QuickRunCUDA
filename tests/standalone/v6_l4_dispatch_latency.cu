// V6 L4: Kernel dispatch latency profiler
// Time from cudaLaunchKernel call to when first thread starts executing
// Use globaltimer (in-kernel) compared to host clock
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

__global__ void capture_start(unsigned long long* dev_t) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        dev_t[0] = t;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long* dev_t;
    cudaMallocManaged(&dev_t, 16 * sizeof(unsigned long long));
    *dev_t = 0;

    int N_RUNS = 100;

    // Warmup
    capture_start<<<1, 32>>>(dev_t);
    cudaDeviceSynchronize();

    // For each launch: read host time before, launch kernel, sync, read host time after,
    // and compare to dev_t (in-kernel timestamp)
    double launch_to_start_ns[100];
    double total_ns[100];

    for (int i = 0; i < N_RUNS; i++) {
        *dev_t = 0;
        auto t_host_before = std::chrono::high_resolution_clock::now();
        unsigned long long host_before_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_host_before.time_since_epoch()).count();

        capture_start<<<1, 32>>>(dev_t);
        cudaDeviceSynchronize();

        auto t_host_after = std::chrono::high_resolution_clock::now();
        unsigned long long host_after_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(t_host_after.time_since_epoch()).count();

        // dev_t is in NANOSECONDS (globaltimer). But it has its own epoch.
        // To compare: dev_t - host_before_ns gives a delta but may be off by clock skew
        unsigned long long dev_ns = *dev_t;
        total_ns[i] = (double)(host_after_ns - host_before_ns);
        // dev_ns ≈ host_before_ns + dispatch_latency
        // If we assume globaltimer epoch ≈ system boot, the delta is meaningful within one run
        launch_to_start_ns[i] = (double)(dev_ns - host_before_ns);  // signed, may be neg
    }

    // Skip first 5 as warmup
    double total_sum = 0, ttf_sum = 0;
    double total_min = 1e18, ttf_min = 1e18;
    double total_max = 0, ttf_max = -1e18;
    for (int i = 5; i < N_RUNS; i++) {
        total_sum += total_ns[i];
        ttf_sum += launch_to_start_ns[i];
        if (total_ns[i] < total_min) total_min = total_ns[i];
        if (total_ns[i] > total_max) total_max = total_ns[i];
        if (launch_to_start_ns[i] < ttf_min) ttf_min = launch_to_start_ns[i];
        if (launch_to_start_ns[i] > ttf_max) ttf_max = launch_to_start_ns[i];
    }
    double n = N_RUNS - 5;

    printf("Kernel dispatch latency:\n");
    printf("  Total launch+exec+sync: avg=%.0f ns  min=%.0f  max=%.0f\n",
           total_sum / n, total_min, total_max);
    printf("  Time to first SM start: avg=%.0f ns  min=%.0f  max=%.0f\n",
           ttf_sum / n, ttf_min, ttf_max);
    printf("  (TTF is approx — depends on globaltimer/host clock skew)\n");

    return 0;
}
