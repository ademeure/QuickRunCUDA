// V10: L1 capacity curve - SEQUENTIAL access (vs Fisher-Yates random)
// Hypothesis: prefetcher allows full L1 capacity for sequential access
#include <cuda_runtime.h>
#include <cstdio>

#define CK(c) do { cudaError_t e=(c); if(e){fprintf(stderr,"%s\n",cudaGetErrorString(e));exit(1);} }while(0)

#define CHAIN_LEN 1024

__global__ void chase_seq(unsigned int* A, unsigned long long* out, int buf_words) {
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

    int sizes_kb[] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 4096};

    printf("=== L1 capacity sweep: SEQUENTIAL chain (stride=32 words = 1 line) ===\n");
    for (int kb : sizes_kb) {
        size_t n = (size_t)kb * 1024 / 4;
        if (n < 32) continue;

        // Sequential chain: A[i] = (i + 32) % n (next cache line, wraps)
        unsigned int* h = (unsigned int*)malloc(n * sizeof(unsigned int));
        for (size_t i = 0; i < n; i++) {
            h[i] = (unsigned int)((i + 32) % n);
        }

        unsigned int* d_A;
        unsigned long long* d_out;
        CK(cudaMalloc(&d_A, n * sizeof(unsigned int)));
        CK(cudaMalloc(&d_out, 16));
        CK(cudaMemcpy(d_A, h, n * sizeof(unsigned int), cudaMemcpyHostToDevice));

        chase_seq<<<1, 32>>>(d_A, d_out, (int)n);
        cudaDeviceSynchronize();
        chase_seq<<<1, 32>>>(d_A, d_out, (int)n);
        cudaDeviceSynchronize();

        unsigned long long cycles;
        cudaMemcpy(&cycles, d_out, 8, cudaMemcpyDeviceToHost);
        printf("  %4d KB: %.1f cy/hop\n", kb, (double)cycles / CHAIN_LEN);

        cudaFree(d_A);
        cudaFree(d_out);
        free(h);
    }
    return 0;
}
