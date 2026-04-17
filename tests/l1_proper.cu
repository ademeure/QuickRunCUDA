// Properly measure effective L1 cache size on B300
// Use chase pattern: each thread does dependent loads, varying working set
#include <cuda_runtime.h>
#include <cstdio>
#include <chrono>

#define ITERS 1024
#define LOOPS 100  // outer loops to amortize startup

// Pointer-chase to measure latency (not BW) — more sensitive to cache hits
extern "C" __global__ void chase(int *src, int *out, int chain_len) {
    int idx = threadIdx.x;
    int sum = 0;
    #pragma unroll 1
    for (int loop = 0; loop < LOOPS; loop++) {
        for (int i = 0; i < chain_len; i++) {
            idx = src[idx];  // dependent load — chain length = chain_len
        }
        sum ^= idx;  // prevent loop hoisting
    }
    if (sum < -1e9) out[blockIdx.x] = sum;
    if (threadIdx.x == 0) out[blockIdx.x] = idx;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);

    int N_max = 1 << 20;  // 4 MB max
    int *h_src = new int[N_max];
    int *d_src, *d_out;
    cudaMalloc(&d_src, N_max * sizeof(int));
    cudaMalloc(&d_out, prop.multiProcessorCount * sizeof(int));

    cudaStream_t s; cudaStreamCreate(&s);

    auto bench_for_size = [&](int N_bytes) {
        // Initialize chase pointers in a region of N_bytes
        int N_words = N_bytes / 4;
        if (N_words < 32) N_words = 32;
        if (N_words > N_max) N_words = N_max;
        // Linked list: each i → next at random within range
        for (int i = 0; i < N_words; i++) {
            h_src[i] = (i + 17) % N_words;  // simple stride
        }
        cudaMemcpy(d_src, h_src, N_words * sizeof(int), cudaMemcpyHostToDevice);

        // Bench
        for (int i = 0; i < 2; i++) {
            chase<<<1, 32, 0, s>>>(d_src, d_out, ITERS);
            cudaDeviceSynchronize();
        }
        float best = 1e30f;
        for (int i = 0; i < 5; i++) {
            auto t0 = std::chrono::high_resolution_clock::now();
            chase<<<1, 32, 0, s>>>(d_src, d_out, ITERS);
            cudaDeviceSynchronize();
            auto t1 = std::chrono::high_resolution_clock::now();
            float ms = std::chrono::duration<float, std::milli>(t1 - t0).count();
            if (ms < best) best = ms;
        }

        // total loads = LOOPS * ITERS per thread (32 threads = same work)
        long long loads = (long long)LOOPS * ITERS;
        float ns_per_load = (best / 1e3f) * 1e9 / loads;
        return ns_per_load;
    };

    printf("# B300 L1 cache size detection (single warp pointer chase)\n");
    printf("# Measure ns per dependent load vs working set size\n");
    printf("# %-12s %-12s\n", "size_KB", "ns_per_load");

    int sizes_kb[] = {1, 2, 4, 8, 16, 32, 48, 64, 96, 128, 256, 512, 1024, 2048, 4096};
    for (int kb : sizes_kb) {
        float ns = bench_for_size(kb * 1024);
        printf("  %-12d %-12.2f\n", kb, ns);
    }

    delete[] h_src;
    cudaStreamDestroy(s);
    cudaFree(d_src); cudaFree(d_out);
    return 0;
}
