// Effect of cudaMemcpyKind on transfer overhead
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);
    cudaStream_t s; cudaStreamCreate(&s);

    void *h_pinned;
    cudaMallocHost(&h_pinned, 64 * 1024 * 1024);
    void *d_buf;
    cudaMalloc(&d_buf, 64 * 1024 * 1024);

    auto bench = [&](auto fn, int trials = 50) {
        for (int i = 0; i < 3; i++) fn();
        cudaDeviceSynchronize();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float us = std::chrono::duration<float, std::micro>(t1-t0).count();
            if (us < best) best = us;
        }
        return best;
    };

    printf("# B300 cudaMemcpyKind effect on H2D transfer (small 4 KB)\n\n");

    {
        float t = bench([&]{
            cudaMemcpyAsync(d_buf, h_pinned, 4096, cudaMemcpyHostToDevice, s);
        });
        printf("  Async HtoD (explicit):  %.2f us\n", t);
    }
    {
        float t = bench([&]{
            cudaMemcpyAsync(d_buf, h_pinned, 4096, cudaMemcpyDefault, s);
        });
        printf("  Async Default (UVA):    %.2f us\n", t);
    }
    {
        float t = bench([&]{
            cudaMemcpy(d_buf, h_pinned, 4096, cudaMemcpyHostToDevice);
        });
        printf("  Sync HtoD (explicit):   %.2f us\n", t);
    }
    {
        float t = bench([&]{
            cudaMemcpy(d_buf, h_pinned, 4096, cudaMemcpyDefault);
        });
        printf("  Sync Default (UVA):     %.2f us\n", t);
    }

    // Test with 1 byte
    printf("\n# 1 byte transfers:\n");
    {
        float t = bench([&]{
            cudaMemcpy(d_buf, h_pinned, 1, cudaMemcpyHostToDevice);
        });
        printf("  Sync 1B HtoD:           %.2f us\n", t);
    }
    {
        float t = bench([&]{
            cudaMemcpy(d_buf, h_pinned, 1, cudaMemcpyDefault);
        });
        printf("  Sync 1B Default:        %.2f us\n", t);
    }

    return 0;
}
