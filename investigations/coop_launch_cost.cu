// Compare cooperative launch vs plain launch overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

extern "C" __global__ void noop() {}

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);

    // Warmup
    for (int i = 0; i < 5; i++) {
        noop<<<1, 1, 0, s>>>();
    }
    cudaDeviceSynchronize();

    auto bench_plain = [&](int trials=20) {
        float best = 1e30f;
        for (int t = 0; t < trials; t++) {
            cudaEventRecord(e0, s);
            noop<<<1, 1, 0, s>>>();
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    auto bench_coop = [&](int trials=20) {
        cudaLaunchAttribute attr;
        attr.id = cudaLaunchAttributeCooperative;
        attr.val.cooperative = 1;
        cudaLaunchConfig_t cfg = {dim3(1), dim3(1), 0, s, &attr, 1};

        float best = 1e30f;
        for (int t = 0; t < trials; t++) {
            cudaEventRecord(e0, s);
            cudaLaunchKernelExC(&cfg, (void*)noop, nullptr);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    auto bench_ex_plain = [&](int trials=20) {
        cudaLaunchConfig_t cfg = {dim3(1), dim3(1), 0, s, nullptr, 0};
        float best = 1e30f;
        for (int t = 0; t < trials; t++) {
            cudaEventRecord(e0, s);
            cudaLaunchKernelExC(&cfg, (void*)noop, nullptr);
            cudaEventRecord(e1, s);
            cudaEventSynchronize(e1);
            float ms; cudaEventElapsedTime(&ms, e0, e1);
            if (ms < best) best = ms;
        }
        return best;
    };

    float t_plain = bench_plain();
    float t_ex = bench_ex_plain();
    float t_coop = bench_coop();

    printf("# B300 cooperative launch vs plain overhead\n");
    printf("  <<<>>> plain:                          %.3f us\n", t_plain*1000);
    printf("  cudaLaunchKernelExC (no attrs):        %.3f us\n", t_ex*1000);
    printf("  cudaLaunchKernelExC (Cooperative=1):   %.3f us (delta %+.3f)\n",
           t_coop*1000, (t_coop-t_ex)*1000);

    return 0;
}
