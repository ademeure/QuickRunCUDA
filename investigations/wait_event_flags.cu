// cudaStreamWaitEvent flag effects
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}
extern "C" __global__ void busy(unsigned long long *out, int cycles) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long t0 = clock64();
        while (clock64() - t0 < cycles) {}
    }
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s1, s2;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

    cudaEvent_t e; cudaEventCreate(&e);
    unsigned long long *d_out; cudaMalloc(&d_out, sizeof(unsigned long long));

    auto bench = [&](auto fn, int trials = 50) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 cudaStreamWaitEvent flag effects\n\n");

    // Default flag
    {
        float t = bench([&]{
            busy<<<1, 32, 0, s1>>>(d_out, 5 * 2032);  // 5us
            cudaEventRecord(e, s1);
            cudaStreamWaitEvent(s2, e, 0);  // default
            noop<<<1, 32, 0, s2>>>();
        });
        printf("  Default flag (0):                  %.2f us\n", t);
    }

    // EventWaitDefault (== 0)
    {
        float t = bench([&]{
            busy<<<1, 32, 0, s1>>>(d_out, 5 * 2032);
            cudaEventRecord(e, s1);
            cudaStreamWaitEvent(s2, e, cudaEventWaitDefault);
            noop<<<1, 32, 0, s2>>>();
        });
        printf("  cudaEventWaitDefault:              %.2f us\n", t);
    }

    // EventWaitExternal
    {
        float t = bench([&]{
            busy<<<1, 32, 0, s1>>>(d_out, 5 * 2032);
            cudaEventRecord(e, s1);
            cudaStreamWaitEvent(s2, e, cudaEventWaitExternal);
            noop<<<1, 32, 0, s2>>>();
        });
        printf("  cudaEventWaitExternal:             %.2f us\n", t);
    }

    return 0;
}
