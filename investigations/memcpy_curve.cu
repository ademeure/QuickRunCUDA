// cudaMemcpy bandwidth curve: size → BW for H2D / D2H / D2D
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    size_t max_size = 256 * 1024 * 1024;
    void *d_src, *d_dst;
    cudaMalloc(&d_src, max_size);
    cudaMalloc(&d_dst, max_size);
    cudaMemset(d_src, 0x40, max_size);

    void *h_pinned;
    cudaMallocHost(&h_pinned, max_size);
    memset(h_pinned, 0x40, max_size);

    void *h_pageable = malloc(max_size);
    memset(h_pageable, 0x40, max_size);

    cudaStream_t s; cudaStreamCreate(&s);

    size_t sizes[] = {
        1024, 4096, 16384, 65536, 256*1024,
        1*1024*1024, 4*1024*1024, 16*1024*1024,
        64*1024*1024, 256*1024*1024
    };

    auto bench = [&](auto fn, int trials=10) {
        for (int i = 0; i < 3; i++) fn();
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

    printf("# B300 cudaMemcpy bandwidth curve\n");
    printf("# %-12s %-14s %-14s %-14s %-14s\n",
           "size_KB", "h2d_pinned", "h2d_pageable", "d2h_pinned", "d2d");

    for (size_t sz : sizes) {
        float bw_h2d_pin = sz / (bench([&]{ cudaMemcpy(d_dst, h_pinned, sz, cudaMemcpyHostToDevice); })/1e3) / 1e9;
        float bw_h2d_pg  = sz / (bench([&]{ cudaMemcpy(d_dst, h_pageable, sz, cudaMemcpyHostToDevice); })/1e3) / 1e9;
        float bw_d2h_pin = sz / (bench([&]{ cudaMemcpy(h_pinned, d_src, sz, cudaMemcpyDeviceToHost); })/1e3) / 1e9;
        float bw_d2d     = sz / (bench([&]{ cudaMemcpy(d_dst, d_src, sz, cudaMemcpyDeviceToDevice); })/1e3) / 1e9;
        printf("  %-12zu %-14.1f %-14.1f %-14.1f %-14.1f\n",
               sz/1024, bw_h2d_pin, bw_h2d_pg, bw_d2h_pin, bw_d2d);
    }

    cudaFree(d_src); cudaFree(d_dst); cudaFreeHost(h_pinned); free(h_pageable);
    return 0;
}
