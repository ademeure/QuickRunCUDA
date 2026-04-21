// V10: True DRAM latency via host-generated Fisher-Yates random permutation
// Defeats any HW prefetcher pattern detection.
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <random>
#include <algorithm>
#include <chrono>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

#define CHAIN_LEN 1024

__global__ void chase(unsigned int* A, unsigned long long* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int idx = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        idx = A[idx];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[1] = (unsigned long long)idx;
}

int main() {
    cudaSetDevice(0);

    // Test multiple buffer sizes
    struct { size_t n; const char* label; } sizes[] = {
        {1024, "4 KB"},
        {64 * 1024, "256 KB"},
        {1024 * 1024, "4 MB"},
        {16 * 1024 * 1024, "64 MB (< L2)"},
        {32 * 1024 * 1024, "128 MB (~L2)"},
        {128 * 1024 * 1024, "512 MB (>> L2, DRAM)"},
        {512 * 1024 * 1024, "2 GB (DRAM)"},
    };

    printf("=== True DRAM latency (host Fisher-Yates random permutation) ===\n");

    std::mt19937 rng(42);

    for (auto& sz : sizes) {
        size_t n = sz.n;
        // Build random permutation on host
        unsigned int* h = (unsigned int*)malloc(n * sizeof(unsigned int));
        for (size_t i = 0; i < n; i++) h[i] = (unsigned int)i;
        // Fisher-Yates shuffle
        for (size_t i = n - 1; i > 0; i--) {
            std::uniform_int_distribution<size_t> dist(0, i - 1);
            size_t j = dist(rng);
            std::swap(h[i], h[j]);
        }
        // Now h is a permutation. Build chain: A[h[i]] = h[(i+1) % n]
        unsigned int* chain = (unsigned int*)malloc(n * sizeof(unsigned int));
        for (size_t i = 0; i < n; i++) {
            chain[h[i]] = h[(i + 1) % n];
        }

        unsigned int* d_A;
        unsigned long long* d_out;
        CK(cudaMalloc(&d_A, n * sizeof(unsigned int)));
        CK(cudaMalloc(&d_out, 16));
        CK(cudaMemcpy(d_A, chain, n * sizeof(unsigned int), cudaMemcpyHostToDevice));

        // Warmup
        chase<<<1, 32>>>(d_A, d_out);
        CK(cudaDeviceSynchronize());

        // Measure
        chase<<<1, 32>>>(d_A, d_out);
        CK(cudaDeviceSynchronize());

        unsigned long long cycles;
        cudaMemcpy(&cycles, d_out, 8, cudaMemcpyDeviceToHost);
        printf("  buf=%-22s: %.1f cy/hop\n", sz.label, (double)cycles / CHAIN_LEN);

        cudaFree(d_A);
        cudaFree(d_out);
        free(h);
        free(chain);
    }
    return 0;
}
