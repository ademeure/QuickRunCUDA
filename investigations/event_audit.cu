// Audit: cudaEventDisableTiming "33% faster" claim
// Use ONLY EventRecord+Sync (no kernel) to isolate event cost
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](unsigned flags, const char *name, int trials = 1000) {
        cudaEvent_t e;
        cudaEventCreateWithFlags(&e, flags);

        for (int i = 0; i < 100; i++) {
            cudaEventRecord(e, s);
            cudaEventSynchronize(e);
        }

        // Just record + sync, no kernel before
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaEventRecord(e, s);
            cudaEventSynchronize(e);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  %-30s %.3f us\n", name, best);

        // Now with noop kernel
        // Warm
        for (int i = 0; i < 100; i++) {
            noop<<<1, 32, 0, s>>>();
            cudaEventRecord(e, s);
            cudaEventSynchronize(e);
        }

        float best2 = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            noop<<<1, 32, 0, s>>>();
            cudaEventRecord(e, s);
            cudaEventSynchronize(e);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best2) best2 = us;
        }
        printf("  %-30s %.3f us  (noop+evRecord+sync)\n", name, best2);

        cudaEventDestroy(e);
    };

    printf("# B300 cudaEventDisableTiming AUDIT\n\n");

    bench(cudaEventDefault, "Default flag");
    bench(cudaEventDisableTiming, "DisableTiming flag");

    return 0;
}
