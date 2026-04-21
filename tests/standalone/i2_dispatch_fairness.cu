// I2: Beyond 128 concurrent kernels - fairness test
// Each kernel records its OWN start time (clock64 from stream)
// Look at distribution of start times for kernels 1..256
// FIFO: kernels 1-128 start first, 129+ start when 1 finishes
// LIFO: kernels 256, 255, ... start first
// Hashed: random subset starts first
#include <cuda_runtime.h>
#include <cstdio>

__global__ void timed_kernel(unsigned long long* start_times, int kid) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t));
        start_times[kid] = t;
        // Spin for ~1 ms so it occupies the SM
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t < 1500000ULL);
    }
}

int main() {
    cudaSetDevice(0);

    int N = 256;
    cudaStream_t streams[256];
    for (int i = 0; i < N; i++) cudaStreamCreate(&streams[i]);

    unsigned long long* start_times;
    cudaMallocManaged(&start_times, N * sizeof(unsigned long long));
    for (int i = 0; i < N; i++) start_times[i] = 0;

    // Launch all 256 in stream order
    cudaDeviceSynchronize();
    for (int i = 0; i < N; i++) {
        timed_kernel<<<1, 32, 0, streams[i]>>>(start_times, i);
    }
    cudaDeviceSynchronize();

    // Print start times relative to min
    unsigned long long min_t = ~0ULL;
    for (int i = 0; i < N; i++) if (start_times[i] < min_t) min_t = start_times[i];

    // Build pairs (kid, rel_start_cy)
    struct P { int kid; unsigned long long rel; };
    P p[256];
    for (int i = 0; i < N; i++) {
        p[i].kid = i;
        p[i].rel = start_times[i] - min_t;
    }
    // Sort by start time
    for (int i = 0; i < N; i++)
        for (int j = i+1; j < N; j++)
            if (p[i].rel > p[j].rel) {
                P t = p[i]; p[i] = p[j]; p[j] = t;
            }

    printf("First 16 kernels to start (kid, rel_start_cy):\n");
    for (int i = 0; i < 16; i++) printf("  rank %d: kid=%d rel_cy=%llu\n", i, p[i].kid, p[i].rel);
    printf("Last 16 kernels to start:\n");
    for (int i = 240; i < 256; i++) printf("  rank %d: kid=%d rel_cy=%llu\n", i, p[i].kid, p[i].rel);

    return 0;
}
