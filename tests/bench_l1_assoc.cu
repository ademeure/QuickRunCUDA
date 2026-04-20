// L1 cache associativity test.
// Walk N distinct addresses with stride S that map to same cache set.
// If N exceeds associativity, evictions cause thrashing.

#ifndef N_LINES
#define N_LINES 4
#endif
#ifndef STRIDE_KB
#define STRIDE_KB 32
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int* p = (unsigned int*)A;
    unsigned int acc = 0;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // Repeated walk over N_LINES addresses spaced by STRIDE_KB
    #pragma unroll 1
    for (int outer = 0; outer < 100; outer++) {
        #pragma unroll
        for (int j = 0; j < N_LINES; j++) {
            unsigned int idx = j * STRIDE_KB * 256 + (unsigned)u2 * acc;  // STRIDE_KB * 1024 / 4 dwords
            unsigned int x;
            asm volatile("ld.global.ca.u32 %0, [%1];"
                         : "=r"(x) : "l"(p + (idx & 0x3FFFFFF)));
            acc ^= x;
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (acc == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = acc;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = 100 * (unsigned long long)N_LINES;
        printf("N_LINES=%d STRIDE=%dKB total_loads=%llu clk=%llu cy/load=%.3f\n",
               N_LINES, STRIDE_KB, total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
