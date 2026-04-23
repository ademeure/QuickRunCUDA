// Audit v2: DRAM cold-line latency via STRIDED permutation chain
// Strategy: chain over CHAIN_NODES nodes spaced by STRIDE bytes through buffer.
// Stride > L2 line ensures each hop is a fresh L2 line.
// CHAIN_NODES * STRIDE > L2 size ensures each hop is a fresh DRAM access.
// We init A[i*STRIDE/4] = ((i+1) % CHAIN_NODES) * STRIDE/4 (sequential ring),
// then optionally shuffle in pairs via Sattolo.
//
// CHAIN_NODES = number of distinct addresses in chain
// STRIDE      = byte distance between consecutive chain nodes

#ifndef CHAIN_NODES
#define CHAIN_NODES (1u<<20)   // 1M nodes
#endif
#ifndef STRIDE_DWORDS
#define STRIDE_DWORDS 64       // 256 B stride (L2 line = 256 B on B300)
#endif
#ifndef CACHE_HINT
#define CACHE_HINT 1           // 0=ca, 1=cg (no L1)
#endif

extern "C" __global__ void init(int* A, float* B, float* C, int u0, int u1, int u2) {
    int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = gridDim.x * blockDim.x;
    unsigned int n = CHAIN_NODES;
    unsigned int s = STRIDE_DWORDS;
    // Step 1: initialize ring A[i*s] = ((i+1) % n) * s
    for (unsigned int i = gtid; i < n; i += total) {
        unsigned int next_idx = ((i + 1u) % n) * s;
        A[i * s] = (int)next_idx;
    }
}

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    int idx = (seed % CHAIN_NODES) * STRIDE_DWORDS;

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
