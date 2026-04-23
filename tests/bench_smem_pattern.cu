// Shared memory access pattern sweep to verify/refute "random = coalesced" claim.
// 32 banks × 4 bytes × 32 lanes = warp-wide access needs all 32 banks hit once for zero conflicts.

#ifndef PATTERN
#define PATTERN 0
#endif
#ifndef ITERS
#define ITERS 10000
#endif

#define SMEM_SIZE 1024  // ints = 32 banks × 32 rows

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    __shared__ int smem[SMEM_SIZE];
    int lane = threadIdx.x;

    // Init smem with non-trivial values
    for (int i = lane; i < SMEM_SIZE; i += 32) {
        smem[i] = i * 1103515245u + 12345u + (unsigned)seed;
    }
    __syncwarp();

    // Pre-compute access index for this lane
    int idx_base;
    #if PATTERN == 0
        idx_base = lane;                    // seq, 32-way unique banks → NO conflicts
    #elif PATTERN == 1
        idx_base = (lane * 2) & 31;         // stride-2 → 2-way conflict
    #elif PATTERN == 2
        idx_base = (lane * 4) & 31;         // stride-4 → 4-way
    #elif PATTERN == 3
        idx_base = (lane * 8) & 31;         // stride-8 → 8-way
    #elif PATTERN == 4
        idx_base = (lane * 16) & 31;        // stride-16 → 16-way
    #elif PATTERN == 5
        idx_base = (lane * 32) & 31;        // stride-32 → 32-way (broadcast same bank)
    #elif PATTERN == 6
        // XORshift "random"
        unsigned int r = (unsigned)(lane + 1);
        r ^= r << 13; r ^= r >> 17; r ^= r << 5;
        idx_base = r & 31;
    #elif PATTERN == 7
        // Golden-ratio hash - well-distributed
        unsigned int r = (unsigned)lane * 2654435769u;
        idx_base = (r >> 27) & 31;
    #elif PATTERN == 8
        idx_base = 0;                        // broadcast same address (bank 0)
    #elif PATTERN == 9
        // Different random hash with seed
        unsigned int r = ((unsigned)lane + (unsigned)seed) * 1013904223u;
        r ^= r >> 15;
        idx_base = r & 31;
    #elif PATTERN == 10
        // Permutation: lane 0→1, 1→2, …, 31→0
        idx_base = (lane + 1) & 31;
    #endif

    int acc = 0;
    unsigned long long t0, t1;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");

    #pragma unroll 16
    for (int i = 0; i < ITERS; i++) {
        // Keep index within 32 banks but rotate across rows by i/32 × 32
        int idx = ((i & 31) * 32) + idx_base;
        idx &= (SMEM_SIZE - 1);
        acc ^= smem[idx];
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

    if (acc == 0xDEADBEEF) C[lane] = (float)acc;

    if (lane == 0) {
        ((unsigned long long*)C)[1024 + blockIdx.x] = t1 - t0;
    }
}
