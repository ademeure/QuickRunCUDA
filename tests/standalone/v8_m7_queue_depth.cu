// V8 M7: HW kernel queue depth — at what point do launches block?
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

    // Launch many kernels; measure when individual launch blocks (back-pressure)
    int delay = 15000000;  // 10 ms each — long enough that queue fills
    int N = 1000;
    double per_launch_us[1000];

    auto t_total_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        slow<<<1, 32, 0, s>>>(buf, delay);
        auto t1 = std::chrono::high_resolution_clock::now();
        per_launch_us[i] = std::chrono::duration<double, std::micro>(t1 - t0).count();
    }
    cudaStreamSynchronize(s);
    auto t_total_end = std::chrono::high_resolution_clock::now();

    double total_us = std::chrono::duration<double, std::micro>(t_total_end - t_total_start).count();

    // Find first launch that blocks (>1 ms)
    int first_block = -1;
    for (int i = 0; i < N; i++) {
        if (per_launch_us[i] > 1000) {  // >1 ms = blocked
            first_block = i;
            break;
        }
    }

    printf("Launched %d kernels (10 ms each).\n", N);
    printf("Total wall: %.0f ms\n", total_us / 1000);
    printf("Per-launch (first 10):\n");
    for (int i = 0; i < 10; i++) printf("  [%d] %.2f us\n", i, per_launch_us[i]);
    printf("First blocked launch (>1 ms): index %d\n", first_block);
    if (first_block > 0) printf("Queue depth ≈ %d kernels\n", first_block);

    return 0;
}
