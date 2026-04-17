// cudaHostAlloc flags: Default vs WriteCombined vs Mapped vs Portable
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>
#include <cstring>

int main() {
    cudaSetDevice(0);

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 2; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    size_t bytes = 256 * 1024 * 1024;
    void *d_buf;
    cudaMalloc(&d_buf, bytes);
    cudaStream_t s; cudaStreamCreate(&s);

    printf("# B300 cudaHostAlloc flag comparison (256 MB H2D and D2H)\n\n");
    printf("# %-25s %-12s %-12s %-15s\n",
           "flags", "h2d_GB/s", "d2h_GB/s", "cpu_write_GB/s");

    struct { const char *name; unsigned flags; } variants[] = {
        {"Default",                cudaHostAllocDefault},
        {"WriteCombined",          cudaHostAllocWriteCombined},
        {"Mapped",                 cudaHostAllocMapped},
        {"Portable",               cudaHostAllocPortable},
        {"WriteCombined+Mapped",   cudaHostAllocWriteCombined | cudaHostAllocMapped},
    };

    for (auto &v : variants) {
        void *p;
        cudaError_t err = cudaHostAlloc(&p, bytes, v.flags);
        if (err) { printf("  %-25s ALLOC_FAIL\n", v.name); continue; }

        // CPU write throughput
        float cpu_t = bench([&]{
            memset(p, 0xab, bytes);
        }, 3);

        float h2d = bench([&]{
            cudaMemcpyAsync(d_buf, p, bytes, cudaMemcpyHostToDevice, s);
            cudaStreamSynchronize(s);
        }, 3);
        float d2h = bench([&]{
            cudaMemcpyAsync(p, d_buf, bytes, cudaMemcpyDeviceToHost, s);
            cudaStreamSynchronize(s);
        }, 3);

        printf("  %-25s %-12.1f %-12.1f %-15.1f\n",
               v.name, bytes/(h2d/1000)/1e9, bytes/(d2h/1000)/1e9, bytes/(cpu_t/1000)/1e9);
        cudaFreeHost(p);
    }

    return 0;
}
