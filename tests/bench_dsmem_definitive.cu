// bench_dsmem_definitive.cu — Definitive DSMEM vs local SMEM latency/throughput
//
// RESOLVES contradiction between §30.H "0.8% overhead" and AUDIT_NOTES "4.7× slower"
//
// Run via QuickRunCUDA with -t 128 -b <cluster_size> and check C[0..7]:
//   C[0] = local SMEM lat total cycles  (from warp 0)
//   C[1] = DSMEM lat total cycles       (from warp 0, block 0 of cluster)
//   C[2] = local SMEM tp total cycles   (ILP=4)
//   C[3] = DSMEM tp total cycles        (ILP=4)
//   C[4..7] = anti-DCE values
//
// Anti-DCE: smem init from arg0 (runtime seed), all outputs unconditional.
// Cluster sync (barrier.cluster) BEFORE timing window — sync overhead excluded.
// Loop bodies verified to contain LDS in the loop (not hoisted).
//
// SASS verification: the mapa output should be in a uniform register (UR),
// giving LDS R, [R+UR] — the correct DSMEM addressing mode in SASS.
//
// Example: ./QuickRunCUDA tests/bench_dsmem_definitive.cu
//          -t 128 -b 4 -0 100 -T 5 --dump-c float_csv

#ifndef CLUSTER_X
#define CLUSTER_X 4
#endif

#define LAT_ITERS 1024
#define TP_ITERS  512
#define SMEM_WORDS 512

extern __shared__ unsigned smem[];

extern "C" __global__ __cluster_dims__(CLUSTER_X, 1, 1) __launch_bounds__(128, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2)
{
    // ── Init local smem from runtime seed (defeats compile-time DCE) ──────────
    for (int i = threadIdx.x; i < SMEM_WORDS; i += blockDim.x) {
        unsigned v = (unsigned)i * 2654435761u ^ (unsigned)seed;
        smem[i] = ((v >> 5) & (SMEM_WORDS - 1u)) * 4u;  // byte offset into smem
    }
    __syncthreads();

    // ── Get cluster position ───────────────────────────────────────────────────
    unsigned my_cta, cluster_size;
    asm volatile("mov.u32 %0, %%cluster_ctaid.x;" : "=r"(my_cta));
    asm volatile("mov.u32 %0, %%cluster_nctaid.x;" : "=r"(cluster_size));

    // ── Cluster barrier BEFORE timing (exclude sync cost from measurements) ───
    asm volatile("barrier.cluster.arrive;" ::: "memory");
    asm volatile("barrier.cluster.wait;"  ::: "memory");

    // Only warp 0 (tid < 32) performs timing measurements
    int tid = threadIdx.x;

    // ── Compute smem addresses ────────────────────────────────────────────────
    unsigned local_base = (unsigned)__cvta_generic_to_shared(&smem[0]);

    // DSMEM: mapa peer's smem[0] — peer_base is warp-uniform (same for all 32 threads)
    // ptxas should promote this to a uniform register (UR), enabling LDS [R+UR] addressing
    unsigned target_cta = (my_cta + 1u) % cluster_size;
    unsigned peer_base;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(peer_base)
                 : "r"(local_base), "r"(target_cta));

    // ── Warp 0: latency measurements ──────────────────────────────────────────
    if (tid < 32) {
        // Initial byte offsets (per-lane starting positions)
        unsigned loc_cur = ((unsigned)(tid * 31 + seed) & (unsigned)(SMEM_WORDS - 1u)) * 4u;
        unsigned rem_cur = ((unsigned)(tid * 31 + seed + 1) & (unsigned)(SMEM_WORDS - 1u)) * 4u;

        unsigned long long t0, t1, t2, t3;

        // Local SMEM latency: dependent chain
        // SASS: LDS loc_cur, [local_base + loc_cur]  (per-thread address changes each iter)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        #pragma unroll 1
        for (int i = 0; i < LAT_ITERS; i++) {
            asm volatile("ld.shared.u32 %0, [%1];"
                         : "=r"(loc_cur) : "r"(local_base + loc_cur) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

        // DSMEM latency: dependent chain using peer smem
        // SASS: LDS rem_cur, [peer_base + rem_cur]  — peer_base should be in UR
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t2) :: "memory");
        #pragma unroll 1
        for (int i = 0; i < LAT_ITERS; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];"
                         : "=r"(rem_cur) : "r"(peer_base + rem_cur) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t3) :: "memory");

        // Write results from thread 0, block 0 only
        if (tid == 0 && blockIdx.x == 0) {
            unsigned long long* Cll = (unsigned long long*)C;
            Cll[0] = t1 - t0;   // local SMEM latency (cycles for LAT_ITERS loads)
            Cll[1] = t3 - t2;   // DSMEM latency
            ((unsigned*)C)[4] = loc_cur;  // anti-DCE
            ((unsigned*)C)[5] = rem_cur;  // anti-DCE
        }
    }

    // All threads participate in throughput (still just warp 0 timing it)
    __syncthreads();

    // ── Warp 0: throughput measurements, ILP=4 ────────────────────────────────
    if (tid < 32) {
        unsigned l0 = ((unsigned)(tid*7+seed+0) & (SMEM_WORDS-1u)) * 4u;
        unsigned l1 = ((unsigned)(tid*7+seed+1) & (SMEM_WORDS-1u)) * 4u;
        unsigned l2 = ((unsigned)(tid*7+seed+2) & (SMEM_WORDS-1u)) * 4u;
        unsigned l3 = ((unsigned)(tid*7+seed+3) & (SMEM_WORDS-1u)) * 4u;
        unsigned r0 = ((unsigned)(tid*7+seed+4) & (SMEM_WORDS-1u)) * 4u;
        unsigned r1 = ((unsigned)(tid*7+seed+5) & (SMEM_WORDS-1u)) * 4u;
        unsigned r2 = ((unsigned)(tid*7+seed+6) & (SMEM_WORDS-1u)) * 4u;
        unsigned r3 = ((unsigned)(tid*7+seed+7) & (SMEM_WORDS-1u)) * 4u;

        unsigned long long t0, t1, t2, t3;

        // Local SMEM throughput (ILP=4 independent chains)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0) :: "memory");
        #pragma unroll 1
        for (int i = 0; i < TP_ITERS; i++) {
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(l0) : "r"(local_base+l0) : "memory");
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(l1) : "r"(local_base+l1) : "memory");
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(l2) : "r"(local_base+l2) : "memory");
            asm volatile("ld.shared.u32 %0, [%1];" : "=r"(l3) : "r"(local_base+l3) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1) :: "memory");

        // DSMEM throughput (ILP=4)
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t2) :: "memory");
        #pragma unroll 1
        for (int i = 0; i < TP_ITERS; i++) {
            asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r0) : "r"(peer_base+r0) : "memory");
            asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r1) : "r"(peer_base+r1) : "memory");
            asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r2) : "r"(peer_base+r2) : "memory");
            asm volatile("ld.shared::cluster.u32 %0, [%1];" : "=r"(r3) : "r"(peer_base+r3) : "memory");
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t3) :: "memory");

        if (tid == 0 && blockIdx.x == 0) {
            unsigned long long* Cll = (unsigned long long*)C;
            Cll[2] = t1 - t0;   // local SMEM throughput
            Cll[3] = t3 - t2;   // DSMEM throughput
            ((unsigned*)C)[6] = l0^l1^l2^l3;
            ((unsigned*)C)[7] = r0^r1^r2^r3;
        }
    }
}
