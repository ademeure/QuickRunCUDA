// G2 RIGOR: stream priority granularity — do all 6 levels behave differently?
//
// THEORETICAL: cudaDeviceGetStreamPriorityRange returns range [-5, 0] = 6 levels
// on B300. Each level should provide different preemption / scheduling priority.
// Test: long low-prio kernel + short high-prio kernel; measure how quickly the
// high-prio kernel STARTS once submitted.

#include <cuda_runtime.h>
#include <cstdio>

__global__ void slow_kernel(unsigned long long *out, int iters) {
    unsigned long long t = clock64();
    unsigned long long acc = 0;
    for (int i = 0; i < iters; i++) acc += clock64() - t + i;
    if (acc == 0xdeadbeef) out[blockIdx.x] = acc;
}

__global__ void fast_kernel(unsigned long long *out) {
    unsigned long long t = clock64();
    if (threadIdx.x == 0 && blockIdx.x == 0) out[0] = t;
}

int main() {
    cudaSetDevice(0);
    int min_p, max_p;
    cudaDeviceGetStreamPriorityRange(&min_p, &max_p);
    printf("# B300 stream priority range: [%d, %d] (%d levels)\n", min_p, max_p, min_p - max_p + 1);

    unsigned long long *d_out; cudaMalloc(&d_out, 1024 * sizeof(unsigned long long));
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);

    auto bench = [&](int low_prio, int high_prio) {
        cudaStream_t s_low, s_high;
        cudaStreamCreateWithPriority(&s_low, cudaStreamNonBlocking, low_prio);
        cudaStreamCreateWithPriority(&s_high, cudaStreamNonBlocking, high_prio);

        // Saturate GPU with slow kernel
        slow_kernel<<<148*8, 256, 0, s_low>>>(d_out, 100000);
        // Immediately launch fast kernel on high-prio stream
        cudaEventRecord(e0, s_high);
        fast_kernel<<<1, 1, 0, s_high>>>(d_out + 100);
        cudaEventRecord(e1, s_high);
        cudaStreamSynchronize(s_high);
        float ms_to_complete; cudaEventElapsedTime(&ms_to_complete, e0, e1);

        cudaStreamSynchronize(s_low);

        cudaStreamDestroy(s_low);
        cudaStreamDestroy(s_high);

        return ms_to_complete;
    };

    printf("# How quickly does fast kernel complete after a slow kernel hogging GPU?\n");
    printf("# slow_prio   fast_prio   fast_complete_ms\n");
    // CUDA returns leastPriority=0, greatestPriority=-5 (lower number = higher priority)
    // Use 0 (lowest pri) for slow stream, sweep high stream from 0 down to -5
    int low = 0;
    for (int high = 0; high >= -5; high--) {
        // Average over a few runs
        float sum = 0;
        int n = 5;
        for (int i = 0; i < n; i++) sum += bench(low, high);
        printf("    %d         %d         %.4f ms\n", low, high, sum/n);
    }

    return 0;
}
