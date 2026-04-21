// V10: prefetch.L2 effectiveness — does the hint actually reduce latency?
#include <cuda_runtime.h>
#include <cstdio>
#include <random>
#include <algorithm>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

#define CHAIN_LEN 512

template<int MODE>
__global__ void chase(unsigned int* A, unsigned long long* out) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    unsigned int idx = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        unsigned int next;
        asm volatile("ld.global.u32 %0, [%1];" : "=r"(next) : "l"(A + idx));
        if (MODE == 1) {
            // Issue prefetch.L2 for NEXT iteration's load target (via next value)
            asm volatile("prefetch.global.L2 [%0];" :: "l"(A + next));
        }
        idx = next;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    out[0] = t1 - t0;
    out[1] = idx;
}

int main() {
    cudaSetDevice(0);
    std::mt19937 rng(42);

    // Buffer in DRAM range (512 MB = 128M unsigned)
    size_t n = 128 * 1024 * 1024;

    unsigned int* h = (unsigned int*)malloc(n * sizeof(unsigned int));
    for (size_t i = 0; i < n; i++) h[i] = (unsigned int)i;
    // Fisher-Yates
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

    // Baseline
    chase<0><<<1, 32>>>(d_A, d_out);
    cudaDeviceSynchronize();
    chase<0><<<1, 32>>>(d_A, d_out);
    cudaDeviceSynchronize();
    unsigned long long cycles_baseline;
    cudaMemcpy(&cycles_baseline, d_out, 8, cudaMemcpyDeviceToHost);

    // With prefetch.L2
    chase<1><<<1, 32>>>(d_A, d_out);
    cudaDeviceSynchronize();
    chase<1><<<1, 32>>>(d_A, d_out);
    cudaDeviceSynchronize();
    unsigned long long cycles_prefetch;
    cudaMemcpy(&cycles_prefetch, d_out, 8, cudaMemcpyDeviceToHost);

    printf("=== prefetch.L2 effectiveness (512 MB random chain) ===\n");
    printf("  Baseline (no prefetch): %.1f cy/hop\n", (double)cycles_baseline / CHAIN_LEN);
    printf("  With prefetch.L2:       %.1f cy/hop\n", (double)cycles_prefetch / CHAIN_LEN);
    printf("  Speedup:                %.2fx\n",
           (double)cycles_baseline / cycles_prefetch);

    cudaFree(d_A); cudaFree(d_out);
    free(h); free(chain);
    return 0;
}
