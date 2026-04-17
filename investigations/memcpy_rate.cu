// Max cudaMemcpyAsync submission rate
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    void *h_pinned;
    cudaMallocHost(&h_pinned, 1024 * 1024);
    void *d_buf;
    cudaMalloc(&d_buf, 1024 * 1024);

    auto bench_async = [&](size_t sz, int trials = 1000) {
        // Warmup
        for (int i = 0; i < 100; i++) cudaMemcpyAsync(d_buf, h_pinned, sz, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < trials; i++) {
            cudaMemcpyAsync(d_buf, h_pinned, sz, cudaMemcpyHostToDevice, s);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        cudaStreamSynchronize(s);
        return std::chrono::duration<float, std::nano>(t1-t0).count() / trials;
    };

    auto bench_sync = [&](size_t sz, int trials = 100) {
        for (int i = 0; i < 5; i++) cudaMemcpy(d_buf, h_pinned, sz, cudaMemcpyHostToDevice);

        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            cudaMemcpy(d_buf, h_pinned, sz, cudaMemcpyHostToDevice);
            auto t1 = std::chrono::high_resolution_clock::now();
            float ns = std::chrono::duration<float, std::nano>(t1-t0).count();
            if (ns < best) best = ns;
        }
        return best;
    };

    printf("# B300 cudaMemcpyAsync submission rate (HtoD pinned)\n\n");
    printf("# %-12s %-15s %-15s\n", "size", "async_ns", "sync_ns");

    for (size_t sz : {1ul, 16ul, 256ul, 4096ul, 65536ul, 1048576ul}) {
        float async_ns = bench_async(sz);
        float sync_ns = bench_sync(sz);
        printf("  %-12zu %-15.0f %-15.0f\n", sz, async_ns, sync_ns);
    }

    return 0;
}
