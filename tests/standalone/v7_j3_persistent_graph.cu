// V7 J3: Persistent kernel in cudaGraph — does it work?
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

    int N_TASKS = 30;

    // Build a graph with the persistent kernel as a single node
    cudaGraph_t graph;
    cudaGraphCreate(&graph, 0);

    cudaKernelNodeParams kp = {};
    kp.func = (void*)persistent_worker;
    kp.gridDim = dim3(1); kp.blockDim = dim3(32);
    void* args[3] = {&dev_mb, &dev_ack, &N_TASKS};
    kp.kernelParams = args;
    cudaGraphNode_t node;
    cudaGraphAddKernelNode(&node, graph, nullptr, 0, &kp);

    cudaGraphExec_t exec;
    cudaError_t err = cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    if (err != cudaSuccess) { printf("Instantiate err: %s\n", cudaGetErrorString(err)); return 1; }

    // Launch the graph (which launches the persistent kernel)
    *mailbox = 0; *ack = 0;
    cudaGraphLaunch(exec, 0);

    std::this_thread::sleep_for(std::chrono::milliseconds(10));

    // Send tasks via mailbox
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int task = 1; task <= N_TASKS; task++) {
        __atomic_store_n(mailbox, task, __ATOMIC_RELEASE);
        while (__atomic_load_n(ack, __ATOMIC_ACQUIRE) != (unsigned)task) {}
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    cudaStreamSynchronize(0);
    double per_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N_TASKS;

    printf("Persistent kernel via cudaGraph: %.2f us/task RTT\n", per_us);
    printf("(Compare V6 E1 direct launch: 2.77 us/task)\n");

    return 0;
}
