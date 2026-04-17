// Stream priority effects on B300
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void busy(unsigned long long *out, int idx, int cycles) {
    if (threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
        out[idx] = clock64();
    }
}

int main() {
    cudaSetDevice(0);

    int prio_lo, prio_hi;
    cudaDeviceGetStreamPriorityRange(&prio_lo, &prio_hi);
    printf("# B300 stream priority range: low=%d, high=%d\n", prio_lo, prio_hi);

    cudaStream_t s_lo, s_hi;
    cudaStreamCreateWithPriority(&s_lo, cudaStreamNonBlocking, prio_lo);
    cudaStreamCreateWithPriority(&s_hi, cudaStreamNonBlocking, prio_hi);

    int prio_check;
    cudaStreamGetPriority(s_lo, &prio_check);
    printf("# Created s_lo with priority: %d\n", prio_check);
    cudaStreamGetPriority(s_hi, &prio_check);
    printf("# Created s_hi with priority: %d\n\n", prio_check);

    unsigned long long *d_out; cudaMalloc(&d_out, 1024*sizeof(unsigned long long));

    // Saturate GPU with low-priority kernels, then launch high-priority
    int delay = 1000 * 2032;  // 1ms each

    printf("# Test: 130 concurrent low-prio kernels + 1 high-prio (each 1ms)\n");
    {
        cudaMemset(d_out, 0, 1024*sizeof(unsigned long long));

        // Launch 130 low-prio kernels (just over the 128 dispatch limit)
        for (int i = 0; i < 130; i++) {
            busy<<<1, 32, 0, s_lo>>>(d_out, i, delay);
        }

        // Launch 1 high-prio kernel
        auto t0 = std::chrono::high_resolution_clock::now();
        busy<<<1, 32, 0, s_hi>>>(d_out, 200, delay);
        cudaStreamSynchronize(s_hi);
        auto t1 = std::chrono::high_resolution_clock::now();
        float us = std::chrono::duration<float, std::micro>(t1-t0).count();

        cudaDeviceSynchronize();
        printf("  High-prio kernel finished in %.0f us\n", us);
        printf("  (single kernel: ~1000 us, queued behind 2 batches: ~2000 us)\n");
    }

    // What about preemption mid-kernel?
    printf("\n# Test: 1 long low-prio kernel (10ms) + 1 hi-prio when SMs full\n");
    {
        cudaMemset(d_out, 0, 1024*sizeof(unsigned long long));

        // Fill all 148 SMs with persistent low-prio work (10ms each)
        for (int i = 0; i < 148; i++) {
            busy<<<1, 32, 0, s_lo>>>(d_out, i, 10000 * 2032);
        }

        cudaDeviceSynchronize();
    }

    return 0;
}
