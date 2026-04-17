// Register spill cost: create pressure via maxregcount or many live vars.
// Compare: no spill (baseline) vs forced spill to local memory.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef NLIVE
#define NLIVE 8
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned tid = threadIdx.x;
    // NLIVE arrays each with fewer vals keeps them live across entire loop
    unsigned v[256];  // big array — will spill if accessed non-statically
    #pragma unroll
    for (int i = 0; i < 256; i++) v[i] = tid + seed + i;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned acc = 0;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // Access v[i % NLIVE] — prevents full unroll, forces local mem
        unsigned idx = i & (NLIVE - 1);
        v[idx] = v[idx] * 1664525u + 1013904223u + seed;
        acc ^= v[idx];
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (tid == 0) {
        ((unsigned long long*)C)[0] = t1-t0;
        C[2] = acc;
    }
}
