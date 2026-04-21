// L7: Driver thread CPU cost under heavy launch load
// Launch many short kernels; measure host CPU time vs wall time
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <sys/resource.h>

__global__ void empty_kernel() {}

int main() {
    // Warmup
    empty_kernel<<<1, 1>>>();
    cudaDeviceSynchronize();

    int N = 100000;

    struct rusage start_usage, end_usage;
    getrusage(RUSAGE_SELF, &start_usage);
    auto start_wall = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < N; i++) {
        empty_kernel<<<1, 1>>>();
    }
    cudaDeviceSynchronize();

    auto end_wall = std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &end_usage);

    double wall_us = std::chrono::duration<double, std::micro>(end_wall - start_wall).count();
    double user_us = (end_usage.ru_utime.tv_sec - start_usage.ru_utime.tv_sec) * 1e6 +
                     (end_usage.ru_utime.tv_usec - start_usage.ru_utime.tv_usec);
    double sys_us = (end_usage.ru_stime.tv_sec - start_usage.ru_stime.tv_sec) * 1e6 +
                    (end_usage.ru_stime.tv_usec - start_usage.ru_stime.tv_usec);

    printf("=== %d empty kernel launches ===\n", N);
    printf("  wall time: %.1f ms\n", wall_us / 1000);
    printf("  user CPU:  %.1f ms (%.1f%%)\n", user_us / 1000, 100.0 * user_us / wall_us);
    printf("  sys CPU:   %.1f ms (%.1f%%)\n", sys_us / 1000, 100.0 * sys_us / wall_us);
    printf("  total CPU: %.1f ms (%.1f%% of wall)\n",
           (user_us + sys_us) / 1000, 100.0 * (user_us + sys_us) / wall_us);
    printf("  per-launch: %.1f ns wall, %.1f ns CPU\n",
           wall_us * 1000 / N, (user_us + sys_us) * 1000 / N);

    return 0;
}
