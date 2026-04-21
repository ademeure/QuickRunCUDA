// V6 F3: Stream + event chain depth latency
// Test how long a chain of event-waits takes:
//   Chain: kernel1 → ev1 → wait → kernel2 → ev2 → wait → ... → kernelN
// Compare N=1, 4, 16, 64, 256 events
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__global__ void noop(unsigned int* buf) {
    if (threadIdx.x == 0 && blockIdx.x == 0) buf[0] = 1;
}

int main() {
    cudaSetDevice(0);
    unsigned int* buf;
    cudaMallocManaged(&buf, 1024);

    int N_VALS[] = {1, 4, 16, 64, 256};
    int N_RUNS = 100;

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    cudaEvent_t evs[256];
    for (int i = 0; i < 256; i++) cudaEventCreateWithFlags(&evs[i], cudaEventDisableTiming);

    cudaEvent_t es, ee;
    cudaEventCreate(&es); cudaEventCreate(&ee);

    // Warmup
    noop<<<1, 32, 0, s1>>>(buf);
    cudaDeviceSynchronize();

    for (int idx = 0; idx < 5; idx++) {
        int N = N_VALS[idx];
        cudaEventRecord(es, 0);
        for (int run = 0; run < N_RUNS; run++) {
            // Chain of N kernels with cross-stream event waits
            for (int i = 0; i < N; i++) {
                noop<<<1, 32, 0, s1>>>(buf);
                cudaEventRecord(evs[i], s1);
                cudaStreamWaitEvent(s2, evs[i], 0);
                noop<<<1, 32, 0, s2>>>(buf);
            }
            cudaStreamSynchronize(s1);
            cudaStreamSynchronize(s2);
        }
        cudaEventRecord(ee, 0);
        cudaEventSynchronize(ee);

        float ms;
        cudaEventElapsedTime(&ms, es, ee);
        double per_chain_us = ms * 1000.0 / N_RUNS;
        double per_event_us = per_chain_us / N;
        printf("Chain N=%-3d: total=%.1f us / chain  per_event=%.2f us\n",
               N, per_chain_us, per_event_us);
    }

    return 0;
}
