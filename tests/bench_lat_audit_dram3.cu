// Audit v3: DRAM latency via Sattolo-shuffled stride chain
// Init: build a single-cycle permutation over CHAIN_NODES positions, each spaced STRIDE_DWORDS apart.
// Single-thread init for clean Sattolo shuffle.
//
// Working set = CHAIN_NODES * STRIDE_DWORDS * 4 bytes
// Each hop visits a unique line; >>L2 forces DRAM cold.

#ifndef CHAIN_NODES
#define CHAIN_NODES (1u<<20)
#endif
#ifndef STRIDE_DWORDS
#define STRIDE_DWORDS 64
#endif
#ifndef CACHE_HINT
#define CACHE_HINT 1   // 0=ca, 1=cg
#endif

extern "C" __global__ void init(int* A, float* B, float* C, int u0, int u1, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int n = CHAIN_NODES;
    unsigned int s = STRIDE_DWORDS;
    // Step 1: ring A[i*s] = ((i+1) % n) * s
    for (unsigned int i = 0; i < n; i++) {
        unsigned int next_idx = ((i + 1u) % n) * s;
        A[i * s] = (int)next_idx;
    }
    // Step 2: Sattolo shuffle (in-place). Pseudo-random swaps.
    unsigned int state = 0x12345678u;
    for (unsigned int i = n - 1; i > 0; i--) {
        // rng
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        unsigned int j = state % i;
        // swap A[i*s] <-> A[j*s]
        int ti = A[i * s];
        int tj = A[j * s];
        A[i * s] = tj;
        A[j * s] = ti;
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    int idx = 0;  // start from A[0]

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if CACHE_HINT == 1
        int next;
        asm volatile("ld.global.cg.b32 %0, [%1];" : "=r"(next) : "l"(A + idx));
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
