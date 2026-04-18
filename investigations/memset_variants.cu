// Test cudaMemset variants and configuration knobs
#include <cuda_runtime.h>
#include <cuda.h>
#include <cstdio>

int main() {
    cudaSetDevice(0);
    cuInit(0);
    cudaEvent_t e0, e1; cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaStream_t s; cudaStreamCreate(&s);

    size_t bytes = 4096ul * 1024 * 1024;
    void *d; cudaMalloc(&d, bytes);

    auto bench = [&](auto fn) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < 7; i++) {
            cudaEventRecord(e0);
            fn();
            cudaEventRecord(e1);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    auto report = [&](float t, const char *name) {
        double bw = bytes/(t/1000)/1e9;
        printf("  %-40s %.3f ms  %.0f GB/s  %.1f%%\n", name, t, bw, bw/7672*100);
    };

    printf("# cudaMemset variants and driver API equivalents (4 GB)\n\n");
    printf("# %-40s %-12s %-12s %-12s\n", "method", "ms", "GB/s", "%peak");

    report(bench([&]{ cudaMemsetAsync(d, 0xab, bytes, s); }), "cudaMemsetAsync(byte)");
    report(bench([&]{ cudaMemset(d, 0xab, bytes); }), "cudaMemset(byte)");
    report(bench([&]{ cuMemsetD8Async((CUdeviceptr)d, 0xab, bytes, s); }), "cuMemsetD8Async");
    // D16: writes 16-bit values
    report(bench([&]{ cuMemsetD16Async((CUdeviceptr)d, 0xabcd, bytes/2, s); }), "cuMemsetD16Async (2-byte)");
    // D32: writes 32-bit values
    report(bench([&]{ cuMemsetD32Async((CUdeviceptr)d, 0xabcdef00, bytes/4, s); }), "cuMemsetD32Async (4-byte)");
    // 2D variants
    report(bench([&]{ cuMemsetD2D8Async((CUdeviceptr)d, bytes/16, 0xab, 16, bytes/16, s); }), "cuMemsetD2D8Async");

    // Sync variants with 0
    report(bench([&]{ cudaMemset(d, 0, bytes); }), "cudaMemset(0) (sync)");

    return 0;
}
