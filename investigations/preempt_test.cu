// Test compute preemption: does high-priority kernel interrupt low-priority?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void long_kernel(int *out, int iters) {
    // Long-running kernel
    float a = 1.0f + threadIdx.x * 0.001f;
    unsigned long long start, end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(start));
    for (int i = 0; i < iters; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(end));
    if (threadIdx.x == 0) {
        out[blockIdx.x * 2] = (int)a;
        out[blockIdx.x * 2 + 1] = (int)(end - start);
    }
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    int sm = prop.multiProcessorCount;

    int preempt;
    cudaDeviceGetAttribute(&preempt, cudaDevAttrComputePreemptionSupported, 0);
    printf("# Compute preemption supported: %d\n", preempt);

    int prio_low, prio_high;
    cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high);
    printf("# Priority range: %d (high) to %d (low)\n\n", prio_high, prio_low);

    int *d_out;
    cudaMalloc(&d_out, sm * 2 * sizeof(int));

    cudaStream_t s_low, s_high;
    cudaStreamCreateWithPriority(&s_low, cudaStreamNonBlocking, prio_low);
    cudaStreamCreateWithPriority(&s_high, cudaStreamNonBlocking, prio_high);

    // Baseline: single low-prio kernel alone
    auto t0 = std::chrono::high_resolution_clock::now();
    long_kernel<<<sm, 128, 0, s_low>>>(d_out, 500000);
    cudaDeviceSynchronize();
    auto t1 = std::chrono::high_resolution_clock::now();
    float ms_alone = std::chrono::duration<float, std::milli>(t1-t0).count();
    printf("Low-prio alone:                       %.3f ms\n", ms_alone);

    // Launch low prio then high prio on DIFFERENT streams
    // On GPU with compute preemption, high-prio kernel should get time
    t0 = std::chrono::high_resolution_clock::now();
    long_kernel<<<sm, 128, 0, s_low>>>(d_out, 500000);
    // Tiny delay — let low-prio actually start
    cudaDeviceSynchronize();  // let low finish then launch high
    auto t_after_low = std::chrono::high_resolution_clock::now();

    long_kernel<<<sm, 128, 0, s_high>>>(d_out, 500000);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();

    float ms_low = std::chrono::duration<float, std::milli>(t_after_low - t0).count();
    float ms_high = std::chrono::duration<float, std::milli>(t1 - t_after_low).count();
    printf("Serial low then high:                 %.3f ms + %.3f ms = %.3f ms\n",
           ms_low, ms_high, ms_low + ms_high);

    // Now try parallel: submit both
    t0 = std::chrono::high_resolution_clock::now();
    long_kernel<<<sm, 128, 0, s_low>>>(d_out, 500000);
    long_kernel<<<sm, 128, 0, s_high>>>(d_out, 500000);
    cudaDeviceSynchronize();
    t1 = std::chrono::high_resolution_clock::now();
    float ms_par = std::chrono::duration<float, std::milli>(t1-t0).count();
    printf("Parallel low+high (no preempt):       %.3f ms (ideal serial = %.3f)\n", ms_par, 2*ms_alone);
    printf("Preemption effect: %+.3f ms (negative = faster, positive = overhead)\n",
           ms_par - 2*ms_alone);

    cudaStreamDestroy(s_low);
    cudaStreamDestroy(s_high);
    cudaFree(d_out);
    return 0;
}
