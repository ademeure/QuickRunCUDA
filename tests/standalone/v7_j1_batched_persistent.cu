// V7 J1: Multi-task batching in persistent kernel
// Process N tasks per signal vs 1 task per signal
// Test if batching reduces per-task overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>

__global__ void persistent_batched(volatile unsigned int* mailbox, volatile unsigned int* ack,
                                    volatile unsigned int* count, int n_signals) {
    if (threadIdx.x != 0) return;
    for (int sig = 1; sig <= n_signals; sig++) {
        while (*mailbox != (unsigned)sig) {}
        // "Process" N tasks (just count them)
        unsigned int n = *count;
        for (int t = 0; t < n; t++) {
            // Tiny per-task work (write + read)
            asm volatile("");  // prevent CSE
        }
        *ack = sig;
        __threadfence_system();
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int *mailbox, *ack, *count;
    cudaHostAlloc((void**)&mailbox, sizeof(unsigned int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&ack, sizeof(unsigned int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&count, sizeof(unsigned int), cudaHostAllocMapped);
    *mailbox = 0; *ack = 0; *count = 1;
    unsigned int *dev_mb, *dev_ack, *dev_count;
    cudaHostGetDevicePointer(&dev_mb, mailbox, 0);
    cudaHostGetDevicePointer(&dev_ack, ack, 0);
    cudaHostGetDevicePointer(&dev_count, count, 0);

    int batch_sizes[] = {1, 4, 16, 64};
    for (int idx = 0; idx < 4; idx++) {
        int batch = batch_sizes[idx];
        *count = batch;
        *mailbox = 0; *ack = 0;

        int N_SIGNALS = 200 / batch;  // total tasks ≈ 200
        persistent_batched<<<1, 32>>>(dev_mb, dev_ack, dev_count, N_SIGNALS);
        std::this_thread::sleep_for(std::chrono::milliseconds(10));

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int sig = 1; sig <= N_SIGNALS; sig++) {
            __atomic_store_n(mailbox, sig, __ATOMIC_RELEASE);
            while (__atomic_load_n(ack, __ATOMIC_ACQUIRE) != (unsigned)sig) {}
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cudaDeviceSynchronize();

        double per_signal_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_SIGNALS;
        double per_task_us = per_signal_us / batch;
        printf("Batch=%-3d signals=%d  signal_RTT=%.2f us  per_task=%.3f us\n",
               batch, N_SIGNALS, per_signal_us, per_task_us);
    }

    return 0;
}
