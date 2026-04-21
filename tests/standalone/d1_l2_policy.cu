// D1: L2 replacement policy — hot+cold pattern to distinguish LRU vs LFU
// Setup: hot_set of N_HOT elements, accessed K times each (high freq)
//        cold_set of N_COLD elements, accessed once (large size)
// Then re-time hot_set access. If LFU keeps hot, latency low. If LRU evicts
// (replaced by cold sweep), latency high.

#include <cuda_runtime.h>
#include <cstdio>

#define N_HOT (256)         // 256 cache lines = 32 KB hot set
#define LINE_SIZE_BYTES 128

// Pointer-chase chain. Each entry points to next.
__global__ void warmup_hot(unsigned int* base, int n) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < n; i++) {
            base[i * (LINE_SIZE_BYTES/4)] = ((i+1) % n) * (LINE_SIZE_BYTES/4);
        }
    }
}

__global__ void chase(unsigned int* base, int n, int iters, unsigned long long* time_out) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int idx = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    for (int i = 0; i < iters; i++) {
        idx = base[idx];
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    *time_out = t1 - t0;
    if (idx == 0xDEADBEEF) base[1] = idx;  // sentinel
}

int main() {
    // Allocate hot region (32 KB) and cold region (256 MB > L2 126 MB)
    size_t hot_bytes = (size_t)N_HOT * LINE_SIZE_BYTES;
    size_t cold_bytes = 256ull * 1024 * 1024;  // 64 MB << L2 (126 MB) — should not evict hot
    int n_cold = cold_bytes / LINE_SIZE_BYTES;

    unsigned int* hot;
    unsigned int* cold;
    cudaMalloc(&hot, hot_bytes);
    cudaMalloc(&cold, cold_bytes);

    unsigned long long* time_dev;
    cudaMalloc(&time_dev, sizeof(unsigned long long));

    // Build pointer chains
    warmup_hot<<<1, 1>>>(hot, N_HOT);
    warmup_hot<<<1, 1>>>(cold, n_cold);
    cudaDeviceSynchronize();

    // Phase 1: warm up hot set (access many times to establish "hotness" if LFU)
    for (int rep = 0; rep < 100; rep++) {
        chase<<<1, 32>>>(hot, N_HOT, N_HOT, time_dev);
    }
    cudaDeviceSynchronize();

    // Time hot access RIGHT after warmup (baseline)
    chase<<<1, 32>>>(hot, N_HOT, 1000, time_dev);
    cudaDeviceSynchronize();
    unsigned long long hot_baseline_cy;
    cudaMemcpy(&hot_baseline_cy, time_dev, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    // Phase 2: sweep cold set ONCE (256 MB > L2)
    chase<<<1, 32>>>(cold, n_cold, n_cold, time_dev);
    cudaDeviceSynchronize();

    // Phase 3: re-time hot set access. Has it been evicted by cold?
    chase<<<1, 32>>>(hot, N_HOT, 1000, time_dev);
    cudaDeviceSynchronize();
    unsigned long long hot_post_cold_cy;
    cudaMemcpy(&hot_post_cold_cy, time_dev, sizeof(unsigned long long), cudaMemcpyDeviceToHost);

    printf("Hot set baseline (before cold sweep): %.1f cy/load (1000 iter)\n",
           (double)hot_baseline_cy / 1000);
    printf("Hot set post-cold sweep:              %.1f cy/load (1000 iter)\n",
           (double)hot_post_cold_cy / 1000);
    printf("Ratio: %.2fx\n", (double)hot_post_cold_cy / hot_baseline_cy);
    printf("\n");
    printf("Interpretation:\n");
    if ((double)hot_post_cold_cy / hot_baseline_cy < 1.3) {
        printf("  Hot set SURVIVED cold sweep → LFU-like behavior or large enough to fit alongside cold\n");
    } else if ((double)hot_post_cold_cy / hot_baseline_cy > 5.0) {
        printf("  Hot set EVICTED by cold sweep → LRU/FIFO behavior (recency matters)\n");
    } else {
        printf("  Mixed (partial eviction) → pseudo-LRU or hash-based\n");
    }

    return 0;
}
