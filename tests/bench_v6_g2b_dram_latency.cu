// V6 G2b: True HBM latency via pointer-chasing (defeats prefetcher)
// Each thread follows a random-shuffled chain through 256 MB buffer
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // Pointer chase: A[i] contains next index
    int idx = threadIdx.x;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        idx = A[idx];  // RAW dependency forces serial fetches
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (idx == -1) C[blockIdx.x] = (float)idx;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("HBM pointer-chase ITERS=%d cy/access=%.2f (true cold-line latency)\n",
               ITERS, (double)(t1-t0)/(double)ITERS);
    }
}

extern "C" __global__ void init(int* A, float* B, float* C, int n_elems, int seed, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    // Initialize A[i] with shuffled indices for pointer-chase
    // Simple LCG-shuffled chain
    for (int i = gtid; i < n_elems; i += total) {
        // Pseudo-random next index (avoids prefetcher pattern)
        unsigned int v = (unsigned int)i;
        v ^= v >> 17; v *= 0xed5ad4bb;
        v ^= v >> 11; v *= 0xac4c1b51;
        v ^= v >> 15; v *= 0x31848bab;
        v ^= v >> 14;
        A[i] = (int)(v & (n_elems - 1));  // mask to valid range
    }
}
