// Verify B300 concurrent kernel slot limit (claimed 128)
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <vector>

extern "C" __global__ void busy_kernel(int *out) {
    // 1 block × 32 threads, ~0.5 ms of work
    float a = 1.0f + threadIdx.x * 0.001f;
    #pragma unroll 1
    for (int i = 0; i < 100000; i++) {
        asm volatile("fma.rn.f32 %0,%0,%1,%2;" : "+f"(a) : "f"(1.0001f), "f"(0.0001f));
    }
    if (threadIdx.x == 0) out[blockIdx.x] = (int)a;
}

int main() {
    cudaSetDevice(0);

    int *d_out;
    cudaMalloc(&d_out, 1024 * sizeof(int));

    auto bench_n = [&](int n) {
        std::vector<cudaStream_t> ss(n);
        for (int i = 0; i < n; i++) cudaStreamCreateWithFlags(&ss[i], cudaStreamNonBlocking);

        // Warmup
        for (int j = 0; j < 3; j++) {
            for (int i = 0; i < n; i++) busy_kernel<<<1, 32, 0, ss[i]>>>(d_out);
            cudaDeviceSynchronize();
        }

        float best = 1e30f;
        for (int t = 0; t < 5; t++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            for (int i = 0; i < n; i++) busy_kernel<<<1, 32, 0, ss[i]>>>(d_out);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }
        for (auto &s : ss) cudaStreamDestroy(s);
        return best;
    };

    float t1 = bench_n(1);
    printf("# B300 concurrent kernel slot verification\n");
    printf("# Single kernel baseline: %.4f ms\n\n", t1);
    printf("# %-6s %-12s %-12s %-12s\n", "N", "time_ms", "effective_N", "slot_ratio");

    // Test around expected boundary
    int sizes[] = {1, 16, 32, 64, 96, 112, 120, 126, 128, 130, 136, 144, 160, 192, 256, 384, 512};
    for (int n : sizes) {
        float t = bench_n(n);
        float effective_n = (n * t1) / t;  // how many ran in parallel
        float slot_ratio = t / t1;
        printf("  %-6d %-12.4f %-12.2f %.2fx\n", n, t, effective_n, slot_ratio);
    }

    cudaFree(d_out);
    return 0;
}
