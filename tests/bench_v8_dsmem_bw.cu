// V8: Cluster DSMEM aggregate bandwidth
// Theoretical: DSMEM goes through SM-to-SM interconnect within GPC.
// On Blackwell, per-link DSMEM BW is less-documented. Let's measure.
//
// Each CTA's warps all read from peer CTA's SMEM. Multiple clusters run on
// different SM groups; sum across all = aggregate GPU DSMEM BW.
//
// Pattern: CTA i in cluster reads from CTA (i+1) mod CLUSTER_X's SMEM.

#ifndef CLUSTER_X
#define CLUSTER_X 8
#endif
#ifndef SMEM_WORDS
#define SMEM_WORDS 2048   // 8 KB per CTA
#endif

// Fixed-size SMEM — extern dynamic didn't get allocated by the harness
// (no --shmem arg path; lead to illegal-memory-access)

extern "C" __global__ __cluster_dims__(CLUSTER_X, 1, 1) __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    __shared__ unsigned int smem[SMEM_WORDS];
    int tid = threadIdx.x;
    int gtid = blockIdx.x * blockDim.x + tid;

    // Init SMEM
    #pragma unroll
    for (int i = tid; i < SMEM_WORDS; i += blockDim.x) {
        smem[i] = ((unsigned)(gtid + i) * 2654435761u) ^ (unsigned)seed;
    }
    __syncthreads();

    // Cluster position
    unsigned my_cta, cluster_size;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_size));

    // Cluster sync before timing
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    // DSMEM peer: next CTA in cluster
    unsigned target_cta = (my_cta + 1u) % cluster_size;
    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base) : "r"(local_base), "r"(target_cta));

    // Sustained DSMEM reads with 8-way ILP — accumulator prevents DCE
    unsigned s0 = 0, s1 = 0, s2 = 0, s3 = 0;
    unsigned s4 = 0, s5 = 0, s6 = 0, s7 = 0;

    unsigned long long t_start, t_end;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_start) :: "memory");

    // SMEM bound: SMEM_WORDS × 4 bytes (8192 for 2048 words).
    // Each iter accesses base..base+224; bound base to 4096 (half).
    int BOUND_MASK = 4095;  // base < 4096, +224 < 4320, fits in 8192 SMEM
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        int base = (tid * 4 + i * 128) & BOUND_MASK;
        unsigned r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r0) : "r"(peer_base + base + 0));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r1) : "r"(peer_base + base + 32));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r2) : "r"(peer_base + base + 64));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r3) : "r"(peer_base + base + 96));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r4) : "r"(peer_base + base + 128));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r5) : "r"(peer_base + base + 160));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r6) : "r"(peer_base + base + 192));
        asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r7) : "r"(peer_base + base + 224));
        s0 += r0; s1 += r1; s2 += r2; s3 += r3;
        s4 += r4; s5 += r5; s6 += r6; s7 += r7;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t_end) :: "memory");

    unsigned long long cycles = t_end - t_start;

    // Anti-DCE: write sum unconditionally (no divergent write with cycles)
    unsigned sum = s0^s1^s2^s3^s4^s5^s6^s7;
    C[gtid] = (float)sum;

    if (tid == 0 && blockIdx.x == 0) {
        // Only 1 thread writes cycles (avoid OOB)
        ((unsigned long long*)C)[1] = cycles;
    }
}
