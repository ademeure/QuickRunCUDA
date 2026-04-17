// cudaEventCreateWithFlags effects
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    auto bench = [&](unsigned flags, const char *name) {
        cudaEvent_t e;
        cudaEventCreateWithFlags(&e, flags);

        // Warmup
        for (int i = 0; i < 5; i++) {
            noop<<<1, 32, 0, s>>>();
            cudaEventRecord(e, s);
            cudaEventSynchronize(e);
        }

        float best = 1e30f;
        for (int i = 0; i < 100; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            noop<<<1, 32, 0, s>>>();
            cudaEventRecord(e, s);
            cudaEventSynchronize(e);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  %-30s %.2f us\n", name, best);
        cudaEventDestroy(e);
    };

    auto bench_create = [&](unsigned flags, const char *name) {
        // Just create + destroy
        for (int i = 0; i < 5; i++) {
            cudaEvent_t e;
            cudaEventCreateWithFlags(&e, flags);
            cudaEventDestroy(e);
        }
        float best = 1e30f;
        for (int i = 0; i < 1000; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaEvent_t e;
            cudaEventCreateWithFlags(&e, flags);
            cudaEventDestroy(e);
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        printf("  %-30s %.3f us\n", name, best);
    };

    printf("# B300 cudaEventCreateWithFlags effects\n\n");

    printf("## Event Create+Destroy cost:\n");
    bench_create(cudaEventDefault, "Default");
    bench_create(cudaEventBlockingSync, "BlockingSync");
    bench_create(cudaEventDisableTiming, "DisableTiming");
    bench_create(cudaEventInterprocess | cudaEventDisableTiming, "Interprocess+NoTiming");

    printf("\n## Noop kernel + Event Record + Sync cost:\n");
    bench(cudaEventDefault, "Default");
    bench(cudaEventBlockingSync, "BlockingSync");
    bench(cudaEventDisableTiming, "DisableTiming");
    bench(cudaEventBlockingSync | cudaEventDisableTiming, "BlockingSync+NoTiming");

    return 0;
}
