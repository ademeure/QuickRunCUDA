// V6 F1: Stream parallelism scaling test
// Run N independent kernels (each ~1 ms) on M streams, measure total time
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void work(unsigned int* buf, int idx, int delay_iters) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        unsigned long long t1;
        do {
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        } while (t1 - t0 < (unsigned long long)delay_iters);
        buf[idx] = (unsigned)idx;
    }
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 1024 * 4);

    int N_STREAMS_VALS[] = {1, 2, 4, 8, 16, 32, 64, 128, 256};
    int delay = 1500000;  // ~1 ms per kernel
    int N_KERNELS = 256;  // enough work for 256 streams

    cudaEvent_t es, ee;
    cudaEventCreate(&es);
    cudaEventCreate(&ee);

    // Warmup
    work<<<1, 32>>>(buf, 0, delay);
    cudaDeviceSynchronize();

    for (int idx = 0; idx < 9; idx++) {
        int N = N_STREAMS_VALS[idx];

        // Create N streams
        cudaStream_t streams[256];
        for (int i = 0; i < N; i++) {
            cudaStreamCreate(&streams[i]);
        }

        cudaEventRecord(es, 0);
        for (int k = 0; k < N_KERNELS; k++) {
            work<<<1, 32, 0, streams[k % N]>>>(buf, k, delay);
        }
        for (int i = 0; i < N; i++) {
            cudaStreamSynchronize(streams[i]);
        }
        cudaEventRecord(ee, 0);
        cudaEventSynchronize(ee);

        float ms;
        cudaEventElapsedTime(&ms, es, ee);
        double per_kernel_us = ms * 1000.0 / N_KERNELS;
        double speedup = (double)N_KERNELS * 1.0 / ms;  // ideal = 1 µs / 1ms = 1
        printf("N_STREAMS=%-2d total=%6.2f ms per_kernel=%6.1f us speedup=%.2fx\n",
               N, ms, per_kernel_us, speedup);

        for (int i = 0; i < N; i++) cudaStreamDestroy(streams[i]);
    }

    return 0;
}
