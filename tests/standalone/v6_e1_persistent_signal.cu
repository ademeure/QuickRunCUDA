// V6 E1: Persistent kernel signal latency
// Persistent kernel polls a mailbox; host writes task IDs; measure response time.
// MODE 0: tight spin (volatile load)
// MODE 1: nanosleep(100) between polls
// MODE 2: nanosleep(1000)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <thread>
#include <cstdlib>

__global__ void persistent_worker(volatile unsigned int* mailbox,
                                   volatile unsigned int* ack,
                                   int n_tasks, int sleep_ns) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    for (int task = 1; task <= n_tasks; task++) {
        // Wait for new task ID to appear in mailbox
        while (true) {
            if (*mailbox == (unsigned)task) break;
            if (sleep_ns > 0) {
                __nanosleep(sleep_ns);
            }
        }
        // Send ack back
        *ack = task;
        __threadfence_system();
    }
}

int main(int argc, char** argv) {
    int sleep_ns = (argc > 1) ? atoi(argv[1]) : 0;
    cudaSetDevice(0);

    int N_TASKS = 100;
    unsigned int *mailbox, *ack;
    cudaHostAlloc((void**)&mailbox, sizeof(unsigned int), cudaHostAllocMapped);
    cudaHostAlloc((void**)&ack, sizeof(unsigned int), cudaHostAllocMapped);
    *mailbox = 0; *ack = 0;
    unsigned int *dev_mailbox, *dev_ack;
    cudaHostGetDevicePointer(&dev_mailbox, mailbox, 0);
    cudaHostGetDevicePointer(&dev_ack, ack, 0);

    // Warmup
    persistent_worker<<<1, 32>>>(dev_mailbox, dev_ack, 1, sleep_ns);
    *mailbox = 1;
    while (*ack != 1) {}
    cudaDeviceSynchronize();
    *mailbox = 0; *ack = 0;

    // Launch persistent kernel
    persistent_worker<<<1, 32>>>(dev_mailbox, dev_ack, N_TASKS, sleep_ns);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Round-trip per task: write mailbox → wait for ack → measure host-side elapsed
    double latencies[100];
    for (int task = 1; task <= N_TASKS; task++) {
        auto t0 = std::chrono::high_resolution_clock::now();
        __atomic_store_n(mailbox, task, __ATOMIC_RELEASE);
        // Spin until kernel acks (with timeout to avoid infinite hang)
        int spin = 0;
        while (__atomic_load_n(ack, __ATOMIC_ACQUIRE) != (unsigned)task) {
            if (++spin > 100000000) {
                fprintf(stderr, "TIMEOUT at task %d (ack=%u)\n", task, *ack);
                exit(1);
            }
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        latencies[task - 1] = std::chrono::duration<double, std::nano>(t1 - t0).count();
    }
    cudaDeviceSynchronize();

    // Skip first 5 as warmup; report stats
    double sum = 0, min_lat = 1e18, max_lat = 0;
    for (int i = 5; i < N_TASKS; i++) {
        sum += latencies[i];
        if (latencies[i] < min_lat) min_lat = latencies[i];
        if (latencies[i] > max_lat) max_lat = latencies[i];
    }
    double avg_lat = sum / (N_TASKS - 5);

    printf("Persistent kernel signal latency (sleep_ns=%d):\n", sleep_ns);
    printf("  avg=%.2f us  min=%.2f us  max=%.2f us\n",
           avg_lat / 1000.0, min_lat / 1000.0, max_lat / 1000.0);

    return 0;
}
