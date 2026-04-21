// V8 E1: cuStreamWaitValue semaphore pattern
// Producer stream writes counter; consumer stream waits for value to reach N
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cuInit(0);
    CUcontext ctx; cuCtxGetCurrent(&ctx);

    unsigned int* sem;
    cudaMallocManaged((void**)&sem, sizeof(unsigned int));
    *sem = 0;
    CUdeviceptr dev_sem = (CUdeviceptr)sem;

    cudaStream_t producer, consumer;
    cudaStreamCreate(&producer);
    cudaStreamCreate(&consumer);

    int N = 100;

    // Warmup
    cuStreamWriteValue32(producer, dev_sem, 1, 0);
    cudaStreamSynchronize(producer);
    *sem = 0;

    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 1; i <= N; i++) {
        // Producer: write semaphore += 1
        cuStreamWriteValue32(producer, dev_sem, (unsigned)i, 0);
        // Consumer: wait until semaphore >= i
        cuStreamWaitValue32(consumer, dev_sem, (unsigned)i, CU_STREAM_WAIT_VALUE_GEQ);
    }
    cudaStreamSynchronize(consumer);
    auto t1 = std::chrono::high_resolution_clock::now();
    double per_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    printf("Producer-consumer semaphore pattern (N=%d):\n", N);
    printf("  Per (write + wait) pair: %.2f us\n", per_us);
    printf("  Final semaphore: %u\n", *sem);

    return 0;
}
