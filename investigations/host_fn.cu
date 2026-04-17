// cudaLaunchHostFunc: how fast does GPU→CPU callback fire?
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <atomic>

extern "C" __global__ void noop() {}

std::atomic<long long> callback_time{0};

void CUDART_CB hostfn(void *user) {
    auto t = std::chrono::high_resolution_clock::now();
    callback_time.store(std::chrono::duration_cast<std::chrono::nanoseconds>(t.time_since_epoch()).count());
}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](auto fn, int trials = 50) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaStreamSynchronize(s);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 cudaLaunchHostFunc latency\n\n");

    // Test 1: Empty hostfn cost
    printf("## Test 1: HostFn alone (no kernel before it)\n");
    {
        float t = bench([&]{
            cudaLaunchHostFunc(s, hostfn, nullptr);
        });
        printf("  HostFn + sync: %.2f us\n", t);
    }

    // Test 2: Kernel + HostFn
    printf("\n## Test 2: Noop kernel → HostFn → noop kernel chain\n");
    {
        float t = bench([&]{
            noop<<<1, 32, 0, s>>>();
            cudaLaunchHostFunc(s, hostfn, nullptr);
            noop<<<1, 32, 0, s>>>();
        });
        printf("  Chain: %.2f us\n", t);
    }

    // Test 3: GPU-to-CPU latency: when does callback fire after kernel completes?
    printf("\n## Test 3: GPU→CPU callback latency\n");
    {
        // Launch kernel, record GPU-side completion via event, then hostfn.
        // Timestamp the callback against the event.
        cudaEvent_t e_done; cudaEventCreate(&e_done);
        float min_lat = 1e30f, max_lat = 0;
        for (int trial = 0; trial < 50; trial++) {
            // Empty stream, synchronously
            cudaDeviceSynchronize();

            callback_time.store(0);
            noop<<<1, 32, 0, s>>>();
            cudaEventRecord(e_done, s);
            cudaLaunchHostFunc(s, hostfn, nullptr);
            cudaStreamSynchronize(s);

            // Calculate latency: callback_time minus when event completed
            // We don't have direct CPU-time of event completion - use cudaEventQuery loop
            // Approximation: assume kernel done immediately, measure callback delay
        }
        // Just report stream sync time as proxy
        float t = bench([&]{
            noop<<<1, 32, 0, s>>>();
            cudaLaunchHostFunc(s, hostfn, nullptr);
        });
        printf("  Noop kernel + HostFn + sync: %.2f us\n", t);
    }

    // Test 4: Multiple HostFns - does CPU thread serialize them?
    printf("\n## Test 4: 100 hostfns vs 1 hostfn\n");
    {
        float t1 = bench([&]{
            cudaLaunchHostFunc(s, hostfn, nullptr);
        });
        float t100 = bench([&]{
            for (int i = 0; i < 100; i++)
                cudaLaunchHostFunc(s, hostfn, nullptr);
        });
        printf("  1 hostfn:     %.2f us\n", t1);
        printf("  100 hostfns:  %.2f us = %.2f us/each\n", t100, t100/100);
    }

    // Test 5: Compare HostFn vs cudaStreamAddCallback (deprecated)
    // Skip - cudaStreamAddCallback is deprecated, HostFn is the replacement

    return 0;
}
