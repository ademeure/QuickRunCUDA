// V9: LDS (shared memory load) latency
// Pointer chase through shared memory. smem init via init kernel OR runtime loop.
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif
#ifndef SMEM_WORDS
#define SMEM_WORDS 1024     // 4 KB
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[SMEM_WORDS];
    if (threadIdx.x != 0) return;  // 1 warp only; measure thread 0

    // Init smem with random permutation (simple LCG)
    const unsigned long long prime = 2654435761ULL;
    unsigned int mask = SMEM_WORDS - 1;
    for (unsigned int i = 0; i < SMEM_WORDS; i++) {
        unsigned int next = (unsigned int)((((unsigned long long)i + 1) * prime) & mask);
        if (next == i) next = (i + 32) & mask;
        smem[i] = next;
    }

    unsigned int idx = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        idx = smem[idx];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[2] = idx;  // Anti-DCE
    }
}
