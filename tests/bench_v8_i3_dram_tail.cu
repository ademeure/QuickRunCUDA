// V8 I3: DRAM refresh impact on tail latency
// Measure per-access latency distribution for random HBM reads
// DRAM refresh (every ~7.8 us for DDR4-like) may cause tail spikes
extern "C" __global__ void init(int* A, float* B, float* C, int n, int seed, int u2) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        for (int i = 0; i < n; i++) A[i] = i;
        unsigned int rng = (unsigned int)seed | 1;
        for (int i = n - 1; i > 0; i--) {
            rng = rng * 1103515245u + 12345u;
            int j = (int)(rng % (unsigned int)(i + 1));
            int tmp = A[i]; A[i] = A[j]; A[j] = tmp;
        }
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Measure per-access latencies, find tail
    int idx = 1;
    unsigned int min_cy = 0xFFFFFFFF, max_cy = 0, sum_cy = 0;

    for (int i = 0; i < ITERS; i++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        idx = A[idx];
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        unsigned int dt = (unsigned int)(t1 - t0);
        if (dt < min_cy) min_cy = dt;
        if (dt > max_cy) max_cy = dt;
        sum_cy += dt;
    }

    if (idx == -1) C[0] = (float)idx;
    printf("HBM pointer-chase tail distribution (%d accesses):\n", ITERS);
    printf("  Min: %u cy\n", min_cy);
    printf("  Avg: %u cy\n", sum_cy / ITERS);
    printf("  Max: %u cy\n", max_cy);
    printf("  Tail/avg ratio: %.2fx\n", (double)max_cy / ((double)sum_cy / ITERS));
}
