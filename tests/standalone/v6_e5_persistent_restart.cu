// V6 E5: Persistent kernel restart cost vs cold launch — simplified
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

__global__ void single_task_kernel(volatile unsigned int* ack) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    *ack = 1;
    __threadfence_system();
}

__global__ void persistent_kernel(volatile unsigned int* mailbox, volatile unsigned int* ack,
                                  int n_tasks) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    for (int task = 1; task <= n_tasks; task++) {
        while (*mailbox != (unsigned)task) {}
        *ack = task;
        __threadfence_system();
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int *mailbox, *ack;
    cudaHostAlloc((void**)&mailbox, sizeof(unsigned int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&ack, sizeof(unsigned int), cudaHostAllocMapped);
    *mailbox = 0; *ack = 0;
    unsigned int *dev_mailbox, *dev_ack;
    cudaHostGetDevicePointer(&dev_mailbox, mailbox, 0);
    cudaHostGetDevicePointer(&dev_ack, ack, 0);

    int N_RUNS = 30;

    // Warmup
    single_task_kernel<<<1, 32>>>(dev_ack);
    cudaDeviceSynchronize();

    // Test 1: Cold-launch each time (no persistent)
    *ack = 0;
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N_RUNS; i++) {
        single_task_kernel<<<1, 32>>>(dev_ack);
        cudaDeviceSynchronize();
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cold_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_RUNS;

    // Test 2: Persistent kernel (single launch, N tasks via signal)
    *mailbox = 0; *ack = 0;
    persistent_kernel<<<1, 32>>>(dev_mailbox, dev_ack, N_RUNS);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    auto t2 = std::chrono::high_resolution_clock::now();
    for (int task = 1; task <= N_RUNS; task++) {
        __atomic_store_n(mailbox, task, __ATOMIC_RELEASE);
        while (__atomic_load_n(ack, __ATOMIC_ACQUIRE) != (unsigned)task) {}
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double persistent_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_RUNS;
    cudaDeviceSynchronize();

    printf("Cold launch per task (kernel + sync): %.2f us\n", cold_us);
    printf("Persistent kernel per task (signal):  %.2f us\n", persistent_us);
    printf("Persistent savings: %.2f us/task (%.1fx faster)\n",
           cold_us - persistent_us, cold_us / persistent_us);

    return 0;
}
