// Decisive test: is cudaMemset DMA (0-SM) or SM-compute?
// If DMA: concurrent FFMA kernel runs in same time as alone
// If SM-compute: concurrent FFMA runs slower (SM contention)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

__launch_bounds__(256, 8) __global__ void ffma(float *out, int iters) {
    float a = threadIdx.x * 0.001f;
    float b = a + 0.001f;
    float c = b + 0.001f;
    float d = c + 0.001f;
    for (int i = 0; i < iters; i++) {
        a = a*1.0001f + 0.0001f;
        b = b*1.0001f + 0.0001f;
        c = c*1.0001f + 0.0001f;
        d = d*1.0001f + 0.0001f;
    }
    if (a+b+c+d < -1e30f) out[blockIdx.x*blockDim.x+threadIdx.x] = a+b+c+d;
}

int main() {
    cudaSetDevice(0);
    float *d_out; cudaMalloc(&d_out, 148 * 256 * sizeof(float));
    void *d_buf; cudaMalloc(&d_buf, 4ull * 1024 * 1024 * 1024);

    cudaStream_t s_ffma, s_memset;
    cudaStreamCreateWithFlags(&s_ffma, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s_memset, cudaStreamNonBlocking);

    cudaEvent_t e0, e1;
    cudaEventCreateWithFlags(&e0, cudaEventDefault);
    cudaEventCreate(&e1);

    int ffma_iters = 1000000;  // ~13 ms FFMA work
    int blocks = 148, threads = 256;

    // Warmup
    for (int i = 0; i < 3; i++) {
        ffma<<<blocks, threads, 0, s_ffma>>>(d_out, ffma_iters);
        cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_memset);
    }
    cudaDeviceSynchronize();

    // Measure FFMA alone (using events on its own stream)
    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 2; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            cudaEventRecord(e0, s_ffma);
            fn();
            cudaEventRecord(e1, s_ffma);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    float ffma_alone = bench([&]{
        ffma<<<blocks, threads, 0, s_ffma>>>(d_out, ffma_iters);
    });

    float ffma_with_memset = bench([&]{
        cudaMemsetAsync(d_buf, 0xab, 4ull*1024*1024*1024, s_memset);
        ffma<<<blocks, threads, 0, s_ffma>>>(d_out, ffma_iters);
    });

    printf("# DMA hypothesis test: is cudaMemset 0-SM?\n\n");
    printf("  FFMA alone:                     %.2f ms\n", ffma_alone);
    printf("  FFMA + concurrent cudaMemset:   %.2f ms\n", ffma_with_memset);
    printf("  Slowdown ratio:                 %.2fx\n", ffma_with_memset / ffma_alone);

    if (ffma_with_memset / ffma_alone < 1.05) {
        printf("  → cudaMemset uses 0 SMs (DMA path)  — HBM contention only\n");
    } else if (ffma_with_memset / ffma_alone < 1.5) {
        printf("  → cudaMemset uses some SMs (partial contention)\n");
    } else {
        printf("  → cudaMemset uses MANY SMs (compute-kernel path)\n");
    }

    return 0;
}
