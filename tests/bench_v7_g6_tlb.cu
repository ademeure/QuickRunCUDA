// V7 G6: TLB / address translation cost for huge buffers
// Sweep buffer size; access random pages; compare per-access latency
// MODE 0: 1 MB buffer (fits in TLB)
// MODE 1: 64 MB buffer (some TLB misses)
// MODE 2: 1 GB buffer (TLB thrashing)
// MODE 3: 8 GB buffer (severe TLB pressure)
extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    // Random access via LCG (defeat sequential prefetcher)
    int idx = 1;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        idx = A[idx];  // Pointer-chase
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (idx == -1) C[blockIdx.x] = (float)idx;

    if (threadIdx.x == 0 && blockIdx.x == 0) {
        printf("ITERS=%d cy/access=%.2f = %.1f ns\n",
               ITERS, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS / 1500.0 * 1000.0);
    }
}

extern "C" __global__ void init(int* A, float* B, float* C, int n_elems, int seed, int u2) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // Fisher-Yates shuffle
        for (int i = 0; i < n_elems; i++) A[i] = i;
        unsigned int rng = (unsigned int)seed | 1;
        for (int i = n_elems - 1; i > 0; i--) {
            rng = rng * 1103515245u + 12345u;
            int j = (int)(rng % (unsigned int)(i + 1));
            int tmp = A[i]; A[i] = A[j]; A[j] = tmp;
        }
    }
}
