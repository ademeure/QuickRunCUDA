// Compare sync method overheads when waiting for finished work
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);
    cudaEvent_t e; cudaEventCreate(&e);

    auto bench = [&](auto fn, int trials = 1000) {
        for (int i = 0; i < 5; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 sync method overhead (already-completed work)\n\n");

    // Warmup: ensure context done
    noop<<<1, 1, 0, s>>>();
    cudaStreamSynchronize(s);

    // Test 1: cudaStreamSynchronize on already-finished stream
    {
        float t = bench([&]{ cudaStreamSynchronize(s); });
        printf("  StreamSynchronize (idle):     %.2f us\n", t);
    }

    // Test 2: cudaStreamQuery
    {
        float t = bench([&]{ cudaStreamQuery(s); });
        printf("  StreamQuery (idle):           %.2f us\n", t);
    }

    // Test 3: cudaEventQuery
    {
        cudaEventRecord(e, s);
        cudaStreamSynchronize(s);
        float t = bench([&]{ cudaEventQuery(e); });
        printf("  EventQuery (recorded+done):   %.2f us\n", t);
    }

    // Test 4: cudaEventSynchronize
    {
        cudaEventRecord(e, s);
        cudaStreamSynchronize(s);
        float t = bench([&]{ cudaEventSynchronize(e); });
        printf("  EventSynchronize (already done): %.2f us\n", t);
    }

    // Test 5: cudaDeviceSynchronize
    {
        float t = bench([&]{ cudaDeviceSynchronize(); });
        printf("  DeviceSynchronize (idle):     %.2f us\n", t);
    }

    // With pending work: kernel + sync method
    printf("\n## With one noop kernel pending:\n");
    {
        float t = bench([&]{
            noop<<<1, 32, 0, s>>>();
            cudaStreamSynchronize(s);
        }, 200);
        printf("  noop + StreamSynchronize:     %.2f us\n", t);
    }
    {
        float t = bench([&]{
            noop<<<1, 32, 0, s>>>();
            cudaEventRecord(e, s);
            cudaEventSynchronize(e);
        }, 200);
        printf("  noop + Event/Synchronize:     %.2f us\n", t);
    }
    {
        float t = bench([&]{
            noop<<<1, 32, 0, s>>>();
            while (cudaStreamQuery(s) != cudaSuccess) {}
        }, 200);
        printf("  noop + spin StreamQuery:      %.2f us\n", t);
    }

    return 0;
}
