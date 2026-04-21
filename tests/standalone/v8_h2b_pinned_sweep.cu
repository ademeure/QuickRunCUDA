// V8 H2b: Pinned vs pageable transfer — WIDE size sweep incl. tiny
// Measure min-latency at 1B / 4B / 16B etc. and full sweep to 1 GB
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    size_t sizes[] = {1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144,
                      1048576, 4194304, 16777216, 67108864, 268435456, 1073741824};
    const char* labels[] = {"1 B", "4 B", "16 B", "64 B", "256 B", "1 KB", "4 KB", "16 KB",
                            "64 KB", "256 KB", "1 MB", "4 MB", "16 MB", "64 MB", "256 MB", "1 GB"};

    void* pinned;
    void* pageable = malloc(1UL << 30);
    void* dev;
    cudaMallocHost(&pinned, 1UL << 30);
    cudaMalloc(&dev, 1UL << 30);
    memset(pinned, 0xAB, 1UL << 30);
    memset(pageable, 0xCD, 1UL << 30);

    cudaStream_t s; cudaStreamCreate(&s);

    printf("Pinned vs Pageable H2D sweep (ASYNC uses pinned path; SYNC for pageable):\n");
    printf("%-10s  %-13s  %-13s  %-13s  %-13s\n",
           "Size", "Pinned async (us)", "Pinned BW", "Pageable (us)", "Pageable BW");

    for (int idx = 0; idx < 16; idx++) {
        size_t sz = sizes[idx];
        int N = (sz < 4096) ? 5000 : (sz < 1048576 ? 1000 : (sz < 16777216 ? 100 : 20));

        // Warmup
        cudaMemcpyAsync(dev, pinned, sz, cudaMemcpyHostToDevice, s);
        cudaStreamSynchronize(s);

        // Pinned ASYNC timing
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            cudaMemcpyAsync(dev, pinned, sz, cudaMemcpyHostToDevice, s);
        }
        cudaStreamSynchronize(s);
        auto t1 = std::chrono::high_resolution_clock::now();
        double pinned_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / N;
        double pinned_bw = sz / (pinned_us / 1e6) / 1e9;

        // Pageable SYNC timing (pageable must use sync path)
        auto t2 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < N; i++) {
            cudaMemcpy(dev, pageable, sz, cudaMemcpyHostToDevice);
        }
        auto t3 = std::chrono::high_resolution_clock::now();
        double pageable_us = std::chrono::duration<double, std::micro>(t3 - t2).count() / N;
        double pageable_bw = sz / (pageable_us / 1e6) / 1e9;

        printf("%-10s  %11.2f  %11.2f GB/s  %11.2f  %11.2f GB/s\n",
               labels[idx], pinned_us, pinned_bw, pageable_us, pageable_bw);
    }

    return 0;
}
