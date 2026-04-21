// V10: L1 cache capacity — find cliff empirically
// Fine-sweep buffer sizes through L1 range (1 KB to 512 KB)
#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <algorithm>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

#define CHAIN_LEN 1024

__global__ void chase(unsigned int* A, unsigned long long* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int idx = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) idx = A[idx];
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[1] = idx;
}

int main() {
    cudaSetDevice(0);
    std::mt19937 rng(42);

    // Fine sweep through L1 range
    int sizes_kb[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 224, 256, 384, 512};

    printf("=== L1 capacity sweep (Fisher-Yates random chain) ===\n");
    for (int kb : sizes_kb) {
        size_t n = (size_t)kb * 1024 / 4;  // words
        if (n < 1) continue;

        unsigned int* h = (unsigned int*)malloc(n * sizeof(unsigned int));
        for (size_t i = 0; i < n; i++) h[i] = (unsigned int)i;
        for (size_t i = n - 1; i > 0; i--) {
            std::uniform_int_distribution<size_t> dist(0, i - 1);
            size_t j = dist(rng);
            std::swap(h[i], h[j]);
        }
        unsigned int* chain = (unsigned int*)malloc(n * sizeof(unsigned int));
        for (size_t i = 0; i < n; i++) chain[h[i]] = h[(i + 1) % n];

        unsigned int* d_A;
        unsigned long long* d_out;
        CK(cudaMalloc(&d_A, n * sizeof(unsigned int)));
        CK(cudaMalloc(&d_out, 16));
        CK(cudaMemcpy(d_A, chain, n * sizeof(unsigned int), cudaMemcpyHostToDevice));

        // Warmup
        chase<<<1, 32>>>(d_A, d_out);
        cudaDeviceSynchronize();

        chase<<<1, 32>>>(d_A, d_out);
        cudaDeviceSynchronize();
        unsigned long long cycles;
        cudaMemcpy(&cycles, d_out, 8, cudaMemcpyDeviceToHost);
        printf("  %4d KB: %.1f cy/hop\n", kb, (double)cycles / CHAIN_LEN);

        cudaFree(d_A);
        cudaFree(d_out);
        free(h);
        free(chain);
    }
    return 0;
}
