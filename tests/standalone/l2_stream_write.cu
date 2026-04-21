// L2: cuStreamWriteValue32 vs kernel-write
// Compare end-to-end latency from CPU enqueue to value visible in managed mem
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void writer(volatile unsigned int* flag, unsigned int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *flag = value;
}

int main() {
    cudaSetDevice(0);
    unsigned int* flag;
    cudaMallocManaged(&flag, sizeof(unsigned int));
    *flag = 0;

    cudaStream_t s;
    cudaStreamCreate(&s);

    int N = 1000;

    // Test 1: kernel-write
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        *flag = 0;
        writer<<<1, 1, 0, s>>>(flag, i + 1);
        while (*flag != (unsigned)(i + 1)) { /* spin */ }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double kernel_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    printf("Kernel-write + CPU spin: %.2f us round-trip\n", kernel_us);

    // Test 2: cuStreamWriteValue32
    CUstream cs = (CUstream)s;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        *flag = 0;
        cuStreamWriteValue32(cs, (CUdeviceptr)flag, (unsigned)(i + 1), 0);
        while (*flag != (unsigned)(i + 1)) { /* spin */ }
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double write_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;
    printf("cuStreamWriteValue32 + CPU spin: %.2f us round-trip\n", write_us);

    // Test 3: pure CPU write (baseline)
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        *flag = (unsigned)(i + 1);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N;
    printf("Pure CPU write to managed mem: %.3f us/write\n", cpu_us);

    return 0;
}
