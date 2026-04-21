// V6 E2: Persistent kernel + mbarrier-based signaling
// Multi-block persistent worker; host signals via global atomic
// Block 0 distributes work to other blocks via global mbarrier-style flags
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

__global__ void persistent_worker(volatile unsigned int* mailbox,
                                  volatile unsigned int* ack,
                                  volatile unsigned int* block_ack,  // per-block ack
                                  int n_tasks, int n_blocks) {
    if (threadIdx.x != 0) return;

    int my_block = blockIdx.x;
    for (int task = 1; task <= n_tasks; task++) {
        // All blocks wait for global signal
        while (*mailbox != (unsigned)task) {}
        // Each block signals its done
        block_ack[my_block] = task;
        __threadfence_system();
        // Block 0 collects acks and signals host
        if (my_block == 0) {
            for (int b = 1; b < n_blocks; b++) {
                while (block_ack[b] != (unsigned)task) {}
            }
            *ack = task;
            __threadfence_system();
        }
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int *mailbox, *ack, *block_ack;
    cudaHostAlloc((void**)&mailbox, sizeof(unsigned int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&ack, sizeof(unsigned int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&block_ack, 256 * sizeof(unsigned int), cudaHostAllocMapped);
    *mailbox = 0; *ack = 0;
    for (int i = 0; i < 256; i++) block_ack[i] = 0;

    unsigned int *dev_mb, *dev_ack, *dev_block_ack;
    cudaHostGetDevicePointer(&dev_mb, mailbox, 0);
    cudaHostGetDevicePointer(&dev_ack, ack, 0);
    cudaHostGetDevicePointer(&dev_block_ack, block_ack, 0);

    int N_TASKS = 30;

    int test_n_blocks[] = {1, 2, 4, 8, 16, 32};
    for (int idx = 0; idx < 6; idx++) {
        int n_blocks = test_n_blocks[idx];
        *mailbox = 0; *ack = 0;
        for (int i = 0; i < n_blocks; i++) block_ack[i] = 0;

        persistent_worker<<<n_blocks, 32>>>(dev_mb, dev_ack, dev_block_ack, N_TASKS, n_blocks);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int task = 1; task <= N_TASKS; task++) {
            __atomic_store_n(mailbox, task, __ATOMIC_RELEASE);
            while (__atomic_load_n(ack, __ATOMIC_ACQUIRE) != (unsigned)task) {}
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();

        double per_task_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TASKS;
        printf("Persistent kernel %d blocks: %.2f us/task RTT\n", n_blocks, per_task_us);
    }

    return 0;
}
