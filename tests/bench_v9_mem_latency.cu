// V9: Memory latency ladder via pointer-chase
// Each node in the chain stores the index of the next node → 1 thread walks it.
// Measure cycles per hop. Vary buffer size to hit L1/L2/DRAM.
#ifndef BUF_WORDS
#define BUF_WORDS 1024       // 4 KB (fits in L1)
#endif
#ifndef CHAIN_LEN
#define CHAIN_LEN 1024
#endif

extern "C" __global__ void init(unsigned int* A, unsigned int* B, unsigned int* C,
                                 int ITERS, int seed, int u2) {
    // Random permutation chain via LCG:
    // A[i] = next_lcg(A[i-1]) mod BUF_WORDS
    // But to form valid chain, use: A[i] = (i * LARGE_PRIME) mod BUF_WORDS
    // This ensures each step jumps unpredictably; prefetcher can't track.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    // Use large prime coprime to 2 (BUF_WORDS is power of 2)
    const unsigned long long prime = 2654435761ULL;  // Knuth's golden ratio mult
    unsigned int mask = BUF_WORDS - 1;
    for (int i = tid; i < BUF_WORDS; i += total) {
        // Next node = (i * prime + offset) & mask, ensuring a valid permutation
        unsigned int next = (unsigned int)((((unsigned long long)i + 1) * prime) & mask);
        // Avoid self-loop: if next == i, skip by +32
        if (next == (unsigned int)i) next = (i + 32) & mask;
        A[i] = next;
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned int* A, unsigned int* B, unsigned int* C,
            int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    // Pointer chase: each load reads A[idx], uses result as next idx
    unsigned int idx = 0;
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
        idx = A[idx];  // Latency chain
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned int*)C)[2] = idx;  // Anti-DCE
    }
}
