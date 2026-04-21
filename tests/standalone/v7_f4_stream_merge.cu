// V7 F4: Stream merge/split patterns
// Test 4 streams running independently then merging via cudaEventRecord/Wait
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void work(unsigned int* buf, int idx, int delay_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do { asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1)); } while (t1 - t0 < (unsigned long long)delay_iters);
        buf[idx] = idx;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 1024);

    cudaStream_t s[4];
    cudaEvent_t e[4], merge;
    for (int i = 0; i < 4; i++) {
        cudaStreamCreateWithFlags(&s[i], cudaStreamNonBlocking);
        cudaEventCreateWithFlags(&e[i], cudaEventDisableTiming);
    }
    cudaEventCreateWithFlags(&merge, cudaEventDisableTiming);

    int delay = 1500000;  // 1 ms
    int N = 100;

    // Warmup
    work<<<1, 32, 0, s[0]>>>(buf, 0, delay);
    cudaDeviceSynchronize();

    // Test 1: 4 parallel streams + cudaStreamSynchronize on each
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 4; k++) work<<<1, 32, 0, s[k]>>>(buf, k, delay);
        for (int k = 0; k < 4; k++) cudaStreamSynchronize(s[k]);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double sync_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;

    // Test 2: 4 parallel streams + event merge into stream 0
    auto t2 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 4; k++) work<<<1, 32, 0, s[k]>>>(buf, k, delay);
        for (int k = 1; k < 4; k++) {
            cudaEventRecord(e[k], s[k]);
            cudaStreamWaitEvent(s[0], e[k], 0);
        }
        cudaStreamSynchronize(s[0]);
    }
    auto t3 = std::chrono::high_resolution_clock::now();
    double merge_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;

    printf("4 streams + sync each:        %.2f us/iter\n", sync_us);
    printf("4 streams + merge via events: %.2f us/iter\n", merge_us);

    return 0;
}
