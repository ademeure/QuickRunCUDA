// V7 M7: GPU↔CPU clock alignment — measure the offset between
// %globaltimer and host steady_clock for accurate TTF (time-to-first-SM)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void capture_globaltimer(unsigned long long* dev_t) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        dev_t[0] = t;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned long long* dev_t;
    cudaMallocManaged(&dev_t, 8 * sizeof(unsigned long long));

    int N = 20;
    long long offsets[20];

    // Warmup
    capture_globaltimer<<<1, 32>>>(dev_t);
    cudaDeviceSynchronize();

    // For each measurement: cudaDeviceSync + capture host & device times in tight sequence
    for (int i = 0; i < N; i++) {
        cudaDeviceSynchronize();
        auto host_before = std::chrono::high_resolution_clock::now();
        unsigned long long host_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
            host_before.time_since_epoch()).count();

        capture_globaltimer<<<1, 32>>>(dev_t);
        cudaDeviceSynchronize();

        // Subtract launch overhead + ~half-RTT to estimate "true" instant
        // Just report device - host (raw offset)
        unsigned long long dev_ns = *dev_t;
        offsets[i] = (long long)dev_ns - (long long)host_ns;
    }

    // Stats
    long long min_off = offsets[0], max_off = offsets[0];
    long long sum = 0;
    for (int i = 0; i < N; i++) {
        sum += offsets[i];
        if (offsets[i] < min_off) min_off = offsets[i];
        if (offsets[i] > max_off) max_off = offsets[i];
    }
    long long avg_off = sum / N;

    printf("GPU globaltimer vs CPU steady_clock:\n");
    printf("  Avg offset: %lld ns (= %.2f sec)\n", avg_off, avg_off / 1e9);
    printf("  Min offset: %lld ns\n", min_off);
    printf("  Max offset: %lld ns\n", max_off);
    printf("  Range:      %lld ns (jitter from launch+exec time)\n", max_off - min_off);
    printf("\nUse min(offset) as alignment ref; subtract from device timestamp\n");
    printf("to get host-equivalent time. Range = TTF lower bound estimate.\n");

    return 0;
}
