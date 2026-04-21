// V7 J2: Persistent kernel + work stealing
// Multiple persistent blocks, each takes work from a global queue
// Block grabs next task ID via atomicAdd; processes; loops
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void worker(unsigned int* counter, unsigned int* done_count, int total_tasks) {
    if (threadIdx.x != 0) return;
    while (true) {
        unsigned int my_task = atomicAdd(counter, 1);
        if (my_task >= (unsigned)total_tasks) break;
        // "Process" task: spin briefly
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do { asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)); } while (t1 - t0 < 1500);
        atomicAdd(done_count, 1);
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int *counter, *done;
    cudaMallocManaged(&counter, 4);
    cudaMallocManaged(&done, 4);

    int N_TASKS = 10000;

    int block_counts[] = {1, 4, 16, 64, 148};
    for (int idx = 0; idx < 5; idx++) {
        int B = block_counts[idx];
        *counter = 0; *done = 0;

        auto t0 = std::chrono::high_resolution_clock::now();
        worker<<<B, 32>>>(counter, done, N_TASKS);
        cudaDeviceSynchronize();
        auto t1 = std::chrono::high_resolution_clock::now();
        double total_us = std::chrono::duration<double, std::micro>(t1 - t0).count();

        printf("Blocks=%-3d done=%u total=%.2f ms (%.2f us/task)\n",
               B, *done, total_us / 1000, total_us / N_TASKS);
    }

    return 0;
}
