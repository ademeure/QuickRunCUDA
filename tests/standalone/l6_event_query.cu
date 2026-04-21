// L6: cudaEventQuery polling overhead
// Compare: cudaEventQuery (poll) vs cudaEventSynchronize (block)
// Also: many polls of completed event
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void short_kernel() {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Spin ~1 ms
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < 1500000ULL);
    }
}

int main() {
    cudaSetDevice(0);
    cudaEvent_t evt;
    cudaEventCreate(&evt);

    // Warmup
    short_kernel<<<1, 32>>>();
    cudaEventRecord(evt);
    cudaEventSynchronize(evt);

    int N = 100000;

    // Test 1: pure cudaEventQuery loop on a COMPLETED event (no kernel running)
    cudaEventSynchronize(evt);  // make sure event done
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < N; i++) {
        cudaEventQuery(evt);
    }
    auto t1 = std::chrono::high_resolution_clock::now();
    double ns_per = std::chrono::duration<double, std::nano>(t1 - t0).count() / N;
    printf("cudaEventQuery on COMPLETED event: %.1f ns/call\n", ns_per);

    // Test 2: cudaEventQuery on PENDING event
    short_kernel<<<1, 32>>>();
    cudaEventRecord(evt);

    int polls = 0;
    auto t2 = std::chrono::high_resolution_clock::now();
    while (cudaEventQuery(evt) != cudaSuccess) polls++;
    auto t3 = std::chrono::high_resolution_clock::now();
    double total_us = std::chrono::duration<double, std::micro>(t3 - t2).count();
    printf("Poll a PENDING event until done: %d polls, %.1f us total = %.1f ns/poll\n",
           polls, total_us, (total_us * 1000.0) / polls);

    // Test 3: cudaEventSynchronize (blocking) baseline
    short_kernel<<<1, 32>>>();
    cudaEventRecord(evt);
    auto t4 = std::chrono::high_resolution_clock::now();
    cudaEventSynchronize(evt);
    auto t5 = std::chrono::high_resolution_clock::now();
    double sync_us = std::chrono::duration<double, std::micro>(t5 - t4).count();
    printf("cudaEventSynchronize (blocks): %.1f us\n", sync_us);

    return 0;
}
