// V6 F2: Stream priority test
// 1. Get priority range
// 2. Launch 128 LOW-priority kernels first
// 3. Launch 1 HIGH-priority kernel after
// 4. Measure when HIGH completes vs queue position
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
    int low, high;
    cudaDeviceGetStreamPriorityRange(&low, &high);
    printf("Stream priority range: low=%d high=%d (lower number = higher priority)\n", low, high);

    unsigned int* buf;
    cudaMallocManaged(&buf, 1024 * 4);

    int delay = 1500000;  // 1 ms
    int N_LOW = 256;  // OVERSUBSCRIBE — exceeds 128 HW slots

    cudaStream_t low_streams[256], high_stream;
    for (int i = 0; i < N_LOW; i++) {
        cudaStreamCreateWithPriority(&low_streams[i], cudaStreamDefault, low);
    }
    cudaStreamCreateWithPriority(&high_stream, cudaStreamDefault, high);

    cudaEvent_t es, ee, e_high;
    cudaEventCreate(&es); cudaEventCreate(&ee); cudaEventCreate(&e_high);

    // Warmup
    work<<<1, 32>>>(buf, 0, delay);
    cudaDeviceSynchronize();

    // Test: launch many low-prio, then one high
    cudaEventRecord(es, 0);
    for (int k = 0; k < N_LOW; k++) {
        work<<<1, 32, 0, low_streams[k]>>>(buf, k, delay);
    }
    work<<<1, 32, 0, high_stream>>>(buf, 9999, delay);
    cudaEventRecord(e_high, high_stream);

    // Wait for everything
    for (int i = 0; i < N_LOW; i++) cudaStreamSynchronize(low_streams[i]);
    cudaStreamSynchronize(high_stream);
    cudaEventRecord(ee, 0);
    cudaEventSynchronize(ee);

    float total_ms, high_ms;
    cudaEventElapsedTime(&total_ms, es, ee);
    cudaEventElapsedTime(&high_ms, es, e_high);

    printf("Low-prio batch (%d kernels): total %.2f ms (would be %.1f ms if serial)\n",
           N_LOW, total_ms, N_LOW * 1.0f);
    printf("High-prio kernel (single): completed at %.2f ms (queue position would imply ~%.0f ms if FIFO)\n",
           high_ms, N_LOW * 1.0f);
    printf("If priority works: high should complete near %.1f ms (1ms each + early dispatch)\n",
           1.0f);

    return 0;
}
