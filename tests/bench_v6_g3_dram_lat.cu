// V6 G3 (revisit G2b): True HBM latency via PROPER pointer-chase
// Buffer is 1 GB (well above L2 = 126 MB)
// Initialize A[i] with TRULY random next index using Fisher-Yates
extern "C" __global__ void init(int* A, float* B, float* C, int n_elems, int seed, int u2) {
    // Single-thread Fisher-Yates shuffle to create perfect random chain
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Initialize identity permutation
        for (int i = 0; i < n_elems; i++) A[i] = i;

        // Fisher-Yates with LCG random
        unsigned int rng = (unsigned int)seed | 1;
        for (int i = n_elems - 1; i > 0; i--) {
            rng = rng * 1103515245u + 12345u;
            int j = (int)(rng % (unsigned int)(i + 1));
            int tmp = A[i]; A[i] = A[j]; A[j] = tmp;
        }
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;  // SINGLE THREAD

    int idx = 7919;  // single thread starts at one offset

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        idx = A[idx];  // pointer chase
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (idx == -1) C[blockIdx.x] = (float)idx;

    printf("HBM pointer-chase (SINGLE THREAD) ITERS=%d cy/access=%.2f = %.1f ns @ 1500 MHz\n",
           ITERS, (double)(t1-t0)/(double)ITERS,
           (double)(t1-t0)/(double)ITERS / 1500.0 * 1000.0);
}
