// cudaHostRegister: pin existing pageable memory
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

int main() {
    cudaSetDevice(0);

    auto bench = [&](auto fn, int trials = 5) {
        for (int i = 0; i < 2; i++) fn();
        float best = 1e30f;
        for (int i = 0; i < trials; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            fn();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1-t0).count();
            if (ms < best) best = ms;
        }
        return best;
    };

    printf("# B300 cudaHostRegister vs cudaMallocHost cost\n\n");
    printf("# %-12s %-15s %-15s %-15s %-15s\n",
           "size_MB", "MallocHost_ms", "Register_ms", "MH_GB/s", "Reg_GB/s");

    for (int sz_mb : {16, 64, 256, 1024}) {
        size_t bytes = (size_t)sz_mb * 1024 * 1024;

        float t_mh = bench([&]{
            void *p; cudaMallocHost(&p, bytes);
            cudaFreeHost(p);
        }, 3);

        void *p = aligned_alloc(4096, bytes);
        memset(p, 0, bytes);

        float t_reg = bench([&]{
            cudaHostRegister(p, bytes, cudaHostRegisterDefault);
            cudaHostUnregister(p);
        }, 3);
        free(p);

        printf("  %-12d %-15.2f %-15.2f %-15.1f %-15.1f\n",
               sz_mb, t_mh, t_reg,
               bytes/(t_mh/1000)/1e9, bytes/(t_reg/1000)/1e9);
    }

    printf("\n## Verify registered mem H2D BW vs pageable\n");
    {
        size_t bytes = 64 * 1024 * 1024;
        void *p = aligned_alloc(4096, bytes); memset(p, 0, bytes);
        void *d; cudaMalloc(&d, bytes);
        cudaStream_t s; cudaStreamCreate(&s);

        float t_unreg = bench([&]{
            cudaMemcpyAsync(d, p, bytes, cudaMemcpyHostToDevice, s);
            cudaStreamSynchronize(s);
        }, 5);

        cudaHostRegister(p, bytes, cudaHostRegisterDefault);
        float t_reg = bench([&]{
            cudaMemcpyAsync(d, p, bytes, cudaMemcpyHostToDevice, s);
            cudaStreamSynchronize(s);
        }, 5);
        cudaHostUnregister(p);

        printf("  Unregistered (pageable) H2D 64 MB: %.2f ms = %.1f GB/s\n",
               t_unreg, bytes/(t_unreg/1000)/1e9);
        printf("  Registered (pinned)     H2D 64 MB: %.2f ms = %.1f GB/s\n",
               t_reg, bytes/(t_reg/1000)/1e9);

        free(p); cudaFree(d); cudaStreamDestroy(s);
    }

    return 0;
}
