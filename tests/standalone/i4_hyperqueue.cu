// I4: Hyperqueue / HW queue count observation
// Launch many small kernels in different streams concurrently
// If HW supports N queues, first N launch in parallel; rest queue
// Measure when concurrency saturates

#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void short_kernel(unsigned int* counter) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        // Spin for ~1 ms (1.5M cy at 1500 MHz)
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < 1500000ULL);
        atomicAdd(counter, 1u);
    }
}

int main() {
    cudaSetDevice(0);

    unsigned int* counter;
    cudaMallocManaged(&counter, sizeof(unsigned int));
    *counter = 0;

    int N_STREAMS = 256;
    cudaStream_t streams[256];
    for (int i = 0; i < N_STREAMS; i++) cudaStreamCreate(&streams[i]);

    // Launch one kernel per stream, all at once
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test for various N (number of streams = max parallel kernels)
    for (int N : {1, 4, 16, 32, 64, 128, 256}) {
        *counter = 0;
        cudaDeviceSynchronize();

        cudaEventRecord(start, 0);
        for (int i = 0; i < N; i++) {
            short_kernel<<<1, 32, 0, streams[i]>>>(counter);
        }
        // Sync all streams
        for (int i = 0; i < N; i++) cudaStreamSynchronize(streams[i]);
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);

        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("N_streams=%3d  total_time=%.3f ms  per-kernel=%.3f ms  speedup=%.2fx (vs sequential %.3f ms)\n",
               N, ms, ms / N, (1.0 * N) / ms, 1.0 * N);
    }

    return 0;
}
