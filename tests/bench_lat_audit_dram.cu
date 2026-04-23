// Audit: True DRAM cold-line latency via pointer-chase
// Init kernel uses fixed buffer size from preprocessor (avoids arg conflict)
#ifndef BUF_WORDS
#define BUF_WORDS (1u << 28)   // 256M dwords = 1 GB default
#endif
#ifndef CACHE_HINT
#define CACHE_HINT 0  // 0=default, 1=cg (no L1), 2=ca (cached all)
#endif

extern "C" __global__ void init(int* A, float* B, float* C, int u0, int u1, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    unsigned int n = BUF_WORDS;
    for (unsigned int i = gtid; i < n; i += total) {
        // Strong scalar hash
        unsigned int v = i;
        v ^= v >> 17; v *= 0xed5ad4bb;
        v ^= v >> 11; v *= 0xac4c1b51;
        v ^= v >> 15; v *= 0x31848bab;
        v ^= v >> 14;
        A[i] = (int)(v & (n - 1));
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    int idx = seed & (BUF_WORDS - 1);

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if CACHE_HINT == 1
        // Bypass L1: ld.global.cg
        int next;
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(next) : "l"(A + idx));
        idx = next;
#elif CACHE_HINT == 2
        // Cache all (default)
        int next;
        asm volatile("ld.global.ca.b32 %0, [%1];" : "=r"(next) : "l"(A + idx));
        idx = next;
#else
        idx = A[idx];
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((int*)C)[2] = idx;
    }
}
