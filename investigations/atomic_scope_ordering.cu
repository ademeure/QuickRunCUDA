// atomic_scope_ordering.cu
// Comprehensive (scope x ordering) cost matrix for B300 atomics.
//
// Design:
//   - 1 block, 32 threads (single warp) — eliminates inter-warp contention
//   - Chain of ITERS atomics per thread, each returns value used as next addr offset
//     (self-dep through the return value to measure true latency)
//   - Both .shared and .global address spaces
//   - All (scope x ordering) combinations valid in PTX
//   - clock64 timing, thread 0 writes result to C[0..1] (u64)
//
// OP encoding (see table below):
//
//  === SHARED MEMORY ===
//   0: atom.shared.add.u32                         (no scope/ordering = relaxed.cta)
//   1: atom.relaxed.cta.shared.add.u32
//   2: atom.relaxed.cluster.shared.add.u32
//   3: atom.relaxed.gpu.shared.add.u32
//   4: atom.relaxed.sys.shared.add.u32
//   5: atom.acquire.cta.shared.add.u32
//   6: atom.acquire.cluster.shared.add.u32
//   7: atom.acquire.gpu.shared.add.u32
//   8: atom.acquire.sys.shared.add.u32
//   9: atom.release.cta.shared.add.u32
//  10: atom.release.cluster.shared.add.u32
//  11: atom.release.gpu.shared.add.u32
//  12: atom.release.sys.shared.add.u32
//  13: atom.acq_rel.cta.shared.add.u32
//  14: atom.acq_rel.cluster.shared.add.u32
//  15: atom.acq_rel.gpu.shared.add.u32
//  16: atom.acq_rel.sys.shared.add.u32
//  17: atom.seq_cst.cta.shared.add.u32
//  18: atom.seq_cst.cluster.shared.add.u32
//  19: atom.seq_cst.gpu.shared.add.u32
//  20: atom.seq_cst.sys.shared.add.u32
//
//  === GLOBAL MEMORY ===
//  21: atom.global.add.u32                         (no scope/ordering = relaxed.gpu)
//  22: atom.relaxed.cta.global.add.u32
//  23: atom.relaxed.cluster.global.add.u32
//  24: atom.relaxed.gpu.global.add.u32
//  25: atom.relaxed.sys.global.add.u32
//  26: atom.acquire.cta.global.add.u32
//  27: atom.acquire.cluster.global.add.u32
//  28: atom.acquire.gpu.global.add.u32
//  29: atom.acquire.sys.global.add.u32
//  30: atom.release.cta.global.add.u32
//  31: atom.release.cluster.global.add.u32
//  32: atom.release.gpu.global.add.u32
//  33: atom.release.sys.global.add.u32
//  34: atom.acq_rel.cta.global.add.u32
//  35: atom.acq_rel.cluster.global.add.u32
//  36: atom.acq_rel.gpu.global.add.u32
//  37: atom.acq_rel.sys.global.add.u32
//  38: atom.seq_cst.cta.global.add.u32
//  39: atom.seq_cst.cluster.global.add.u32
//  40: atom.seq_cst.gpu.global.add.u32
//  41: atom.seq_cst.sys.global.add.u32

#ifndef OP
#define OP 0
#endif

#ifndef ITERS
#define ITERS 1024
#endif

extern __shared__ unsigned smem[];

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int iters, int seed, int u2) {
    // Initialize shared memory
    if (threadIdx.x < 256) smem[threadIdx.x] = threadIdx.x ^ (unsigned)seed;
    __syncthreads();

    // Each thread works on its own location to avoid cross-thread serialization
    // Thread t -> slot t (shared), slot A[t] (global)
    unsigned saddr = (unsigned)__cvta_generic_to_shared(&smem[threadIdx.x]);
    unsigned long long gaddr = (unsigned long long)A + (unsigned long long)threadIdx.x * 4ULL;

    unsigned v = (unsigned)seed + threadIdx.x;
    unsigned long long t0, t1;

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned r;

        // === SHARED: no qualifier (baseline) ===
#if OP == 0
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));

        // === SHARED: relaxed ===
#elif OP == 1
        asm volatile("atom.relaxed.cta.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 2
        asm volatile("atom.relaxed.cluster.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 3
        asm volatile("atom.relaxed.gpu.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 4
        asm volatile("atom.relaxed.sys.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));

        // === SHARED: acquire ===
#elif OP == 5
        asm volatile("atom.acquire.cta.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 6
        asm volatile("atom.acquire.cluster.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 7
        asm volatile("atom.acquire.gpu.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 8
        asm volatile("atom.acquire.sys.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));

        // === SHARED: release ===
#elif OP == 9
        asm volatile("atom.release.cta.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 10
        asm volatile("atom.release.cluster.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 11
        asm volatile("atom.release.gpu.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 12
        asm volatile("atom.release.sys.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));

        // === SHARED: acq_rel ===
#elif OP == 13
        asm volatile("atom.acq_rel.cta.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 14
        asm volatile("atom.acq_rel.cluster.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 15
        asm volatile("atom.acq_rel.gpu.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 16
        asm volatile("atom.acq_rel.sys.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));

        // === SHARED: seq_cst ===
#elif OP == 17
        asm volatile("atom.seq_cst.cta.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 18
        asm volatile("atom.seq_cst.cluster.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 19
        asm volatile("atom.seq_cst.gpu.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));
#elif OP == 20
        asm volatile("atom.seq_cst.sys.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(saddr), "r"(v));

        // === GLOBAL: no qualifier (baseline) ===
#elif OP == 21
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));

        // === GLOBAL: relaxed ===
#elif OP == 22
        asm volatile("atom.relaxed.cta.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 23
        asm volatile("atom.relaxed.cluster.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 24
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 25
        asm volatile("atom.relaxed.sys.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));

        // === GLOBAL: acquire ===
#elif OP == 26
        asm volatile("atom.acquire.cta.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 27
        asm volatile("atom.acquire.cluster.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 28
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 29
        asm volatile("atom.acquire.sys.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));

        // === GLOBAL: release ===
#elif OP == 30
        asm volatile("atom.release.cta.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 31
        asm volatile("atom.release.cluster.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 32
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 33
        asm volatile("atom.release.sys.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));

        // === GLOBAL: acq_rel ===
#elif OP == 34
        asm volatile("atom.acq_rel.cta.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 35
        asm volatile("atom.acq_rel.cluster.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 36
        asm volatile("atom.acq_rel.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 37
        asm volatile("atom.acq_rel.sys.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));

        // === GLOBAL: seq_cst ===
#elif OP == 38
        asm volatile("atom.seq_cst.cta.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 39
        asm volatile("atom.seq_cst.cluster.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 40
        asm volatile("atom.seq_cst.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#elif OP == 41
        asm volatile("atom.seq_cst.sys.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(gaddr), "r"(v));
#else
#error "Unknown OP"
#endif

        // Chain through return value (prevents DCE and makes it a true latency chain)
        v = r + 1u;
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    // Thread 0 writes timing; all threads write v to defeat DCE
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
    }
    // Anti-DCE: all threads store their final v value
    C[64 + threadIdx.x] = v;
}
