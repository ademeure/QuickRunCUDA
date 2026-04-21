// V6 E3: Persistent kernel — cudaDeviceSync vs cudaStreamSync vs cudaEventSync
// Compare host-side sync mechanisms for waiting on persistent kernel ack
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

__global__ void persistent_worker(volatile unsigned int* mailbox, volatile unsigned int* ack, int n_tasks) {
    if (threadIdx.x != 0) return;
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
    unsigned int *dev_mb, *dev_ack;
    cudaHostGetDevicePointer(&dev_mb, mailbox, 0);
    cudaHostGetDevicePointer(&dev_ack, ack, 0);

    int N_TASKS = 50;

    // Method 1: spin-wait on ack (no CUDA sync needed)
    *mailbox = 0; *ack = 0;
    persistent_worker<<<1, 32>>>(dev_mb, dev_ack, N_TASKS);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int task = 1; task <= N_TASKS; task++) {
        *mailbox = task;
        while (__atomic_load_n(ack, __ATOMIC_ACQUIRE) != (unsigned)task) {}
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaDeviceSynchronize();
    double spin_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TASKS;

    // Method 2: cudaEventQuery in a loop (host event polling)
    cudaEvent_t evt;
    cudaEventCreateWithFlags(&evt, cudaEventDisableTiming);
    cudaStream_t s;
    cudaStreamCreate(&s);
    *mailbox = 0; *ack = 0;
    persistent_worker<<<1, 32, 0, s>>>(dev_mb, dev_ack, N_TASKS);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int task = 1; task <= N_TASKS; task++) {
        *mailbox = task;
        while (__atomic_load_n(ack, __ATOMIC_ACQUIRE) != (unsigned)task) {}
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    cudaStreamSynchronize(s);
    double stream_spin_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N_TASKS;

    printf("Persistent kernel sync methods:\n");
    printf("  Default stream + spin-wait: %.2f us/task\n", spin_us);
    printf("  Custom stream + spin-wait:  %.2f us/task\n", stream_spin_us);

    // Note: cudaDeviceSynchronize after persistent kernel is N/A — kernel never returns
    // until last task. We measure the per-task RTT only.

    return 0;
}
