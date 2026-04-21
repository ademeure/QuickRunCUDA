// L1: cuStreamWaitValue32 latency vs CPU-side spin
// Setup: GPU writes a flag value at end of kernel. CPU waits for it.
// MODE 0: cudaStreamWaitValue32 + WriteValue32 chain
// MODE 1: CPU spin on managed memory pointer
// Compare end-to-end host-side latency
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>
#include <chrono>

__global__ void writer_kernel(volatile unsigned int* flag, unsigned int value) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Spin briefly so CPU has time to set up wait
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < 1500ULL);  // ~1 us
        *flag = value;
    }
}

int main() {
    cudaSetDevice(0);

    unsigned int* flag;
    cudaMallocManaged(&flag, sizeof(unsigned int));
    *flag = 0;

    cudaStream_t s;
    cudaStreamCreate(&s);

    int N = 1000;

    // Test 1: CPU-side spin on managed flag
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        *flag = 0;
        unsigned int target = i + 1;
        writer_kernel<<<1, 32, 0, s>>>(flag, target);
        // CPU spin
        while (*flag != target) { /* spin */ }
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double cpu_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
    printf("CPU-side spin on managed flag: %.2f us/round-trip\n", cpu_us);

    // Test 2: cuStreamWaitValue32 (GPU waits, CPU just records event)
    CUstream cs = (CUstream)s;
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        *flag = 0;
        unsigned int target = i + 1;
        writer_kernel<<<1, 32, 0, s>>>(flag, target);
        // Wait via stream API
        cuStreamWaitValue32(cs, (CUdeviceptr)flag, target, CU_STREAM_WAIT_VALUE_EQ);
        cudaStreamSynchronize(s);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double waitval_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;
    printf("cuStreamWaitValue32 + sync: %.2f us/round-trip\n", waitval_us);

    // Test 3: cudaStreamSynchronize alone (baseline)
    auto t4 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        *flag = 0;
        unsigned int target = i + 1;
        writer_kernel<<<1, 32, 0, s>>>(flag, target);
        cudaStreamSynchronize(s);
    }
    auto t5 = std::chrono::high_resolution_clock::now();
    double sync_us = std::chrono::duration<double, std::micro>(t5 - t4).count() / N;
    printf("cudaStreamSynchronize only: %.2f us/round-trip\n", sync_us);

    return 0;
}
