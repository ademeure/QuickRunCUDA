// atomic_peak.cu — Comprehensive global-memory atomic throughput characterization
//
// Tests atomicAdd on global memory with:
//   MODE 0: full contention (all threads → same address)
//   MODE 1: per-warp contention (32-way: one address per warp)
//   MODE 2: no contention, no coalescing (per-thread address, stride=256B)
//   MODE 3: no contention, coalesced stride=4B (adjacent threads share 32B block → 8:1)
//   MODE 4: no contention, stride=8B (4:1 coalesce)
//   MODE 5: no contention, stride=16B (2:1 coalesce)
//   MODE 6: no contention, stride=32B (1:1 coalesce, 1 sector apart)
//   MODE 7: no contention, stride=64B (1:1 coalesce, 1 cache-line apart)
//   MODE 8: no contention, stride=128B
//   MODE 9: per-thread but relaxed+gpu scope (atom.relaxed.gpu.global)
//   MODE 10: per-thread u64 coalesced stride=8B
//   MODE 11: per-thread u64 uncoalesced stride=256B
//   MODE 12: per-thread atomicCAS u32 coalesced stride=4B
//   MODE 13: per-thread red.global (no return value, fire-and-forget)
//   MODE 14: per-thread atom.acquire.gpu scope (ordering cost test)
//   MODE 15: per-thread atom.release.gpu scope (ordering cost test)
//
// UNROLL controls how many atoms are issued per loop iteration (default 16).
// Addresses are computed so no thread ever revisits the same location across
// iterations (use enough buffer space via -A).

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef MODE
#define MODE 2
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 1024
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int* arr = (unsigned int*)A;
    unsigned long long gtid = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;

    // Working accumulators to prevent DCE
    unsigned int v0 = seed, v1 = seed+1, v2 = seed+2, v3 = seed+3;
    unsigned int v4 = seed+4, v5 = seed+5, v6 = seed+6, v7 = seed+7;
    unsigned int v8 = seed+8, v9 = seed+9, v10 = seed+10, v11 = seed+11;
    unsigned int v12 = seed+12, v13 = seed+13, v14 = seed+14, v15 = seed+15;

    // ---- Address computation per mode ----
    // For unrolled chains: each chain j operates at base + j * stride_between_chains
    // stride_between_chains is chosen large enough that no two chains share a 32B block
    // for non-contention modes.

#if MODE == 0
    // All threads → address 0 (maximum contention)
#define ADR(j) (unsigned long long)arr  // same address for all
    unsigned long long addr = (unsigned long long)arr;

#elif MODE == 1
    // Per-warp unique address (32-way contention within warp's slot)
    unsigned int warpid = (unsigned int)(gtid / 32);
    unsigned long long addr = (unsigned long long)(arr + warpid * 8ULL);
#define ADR(j) (addr + (unsigned long long)(j) * 32ULL)

#elif MODE == 2
    // Per-thread, stride=256B (no coalescing, no contention)
    // Each thread's chain j at offset: gtid * 4096 + j * 256
    unsigned long long addr = (unsigned long long)(arr) + gtid * 4096ULL;
#define ADR(j) (addr + (unsigned long long)(j) * 256ULL)

#elif MODE == 3
    // Per-thread, stride=4B — adjacent threads share 32B blocks (8:1 coalesce)
    // Chain j: thread gtid hits slot gtid + j * (total_threads), so all threads
    // in a warp hit consecutive 4B slots in the same 32B block.
    // Layout: [t0_j0][t1_j0]...[t31_j0][t0_j1]...  each chain is 128B (32 threads × 4B)
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 4ULL)

#elif MODE == 4
    // Per-thread, stride=8B (4:1 coalesce, adjacent pairs share 32B block)
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 8ULL)

#elif MODE == 5
    // Per-thread, stride=16B (2:1 coalesce)
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 16ULL)

#elif MODE == 6
    // Per-thread, stride=32B (1:1 coalesce, one sector per thread)
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 32ULL)

#elif MODE == 7
    // Per-thread, stride=64B (1 cache-line per thread, no coalescing)
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 64ULL)

#elif MODE == 8
    // Per-thread, stride=128B
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 128ULL)

#elif MODE == 9
    // Per-thread, stride=256B, relaxed.gpu scope (no ordering overhead)
    unsigned long long addr = (unsigned long long)(arr) + gtid * 4096ULL;
#define ADR(j) (addr + (unsigned long long)(j) * 256ULL)

#elif MODE == 10
    // Per-thread u64, stride=8B (4:1 coalesce for 8B slots)
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 8ULL)

#elif MODE == 11
    // Per-thread u64, stride=256B (no coalescing)
    unsigned long long addr = (unsigned long long)(arr) + gtid * 4096ULL;
#define ADR(j) (addr + (unsigned long long)(j) * 256ULL)

#elif MODE == 12
    // Per-thread atomicCAS u32, stride=4B (coalesced, 8:1)
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 4ULL)

#elif MODE == 13
    // Per-thread red.global (no return), stride=4B coalesced
    unsigned long long total_threads = (unsigned long long)gridDim.x * blockDim.x;
#define ADR(j) ((unsigned long long)arr + (gtid + (unsigned long long)(j) * total_threads) * 4ULL)

#elif MODE == 14
    // Per-thread atom.acquire.gpu — does ordering add cost?
    unsigned long long addr = (unsigned long long)(arr) + gtid * 4096ULL;
#define ADR(j) (addr + (unsigned long long)(j) * 256ULL)

#elif MODE == 15
    // Per-thread atom.release.gpu — does ordering add cost?
    unsigned long long addr = (unsigned long long)(arr) + gtid * 4096ULL;
#define ADR(j) (addr + (unsigned long long)(j) * 256ULL)
#endif

    // ---- Main loop ----
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Contention: all threads hit addr, accumulate into v0
        unsigned int r;
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r) : "l"(addr), "r"(1u));
        v0 ^= r;
#elif MODE == 1
        // Per-warp contention: 8 independent chains within warp's sector
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r0) : "l"(ADR(0)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r1) : "l"(ADR(1)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r2) : "l"(ADR(2)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r3) : "l"(ADR(3)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r4) : "l"(ADR(4)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r5) : "l"(ADR(5)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r6) : "l"(ADR(6)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r7) : "l"(ADR(7)), "r"(1u));
        v0 ^= r0^r1^r2^r3^r4^r5^r6^r7;
#elif MODE == 2 || MODE == 7 || MODE == 8
        // Per-thread uncoalesced: UNROLL independent chains
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
        unsigned int r8, r9, r10, r11, r12, r13, r14, r15;
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r0) : "l"(ADR(0)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r1) : "l"(ADR(1)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r2) : "l"(ADR(2)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r3) : "l"(ADR(3)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r4) : "l"(ADR(4)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r5) : "l"(ADR(5)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r6) : "l"(ADR(6)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r7) : "l"(ADR(7)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r8) : "l"(ADR(8)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r9) : "l"(ADR(9)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r10) : "l"(ADR(10)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r11) : "l"(ADR(11)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r12) : "l"(ADR(12)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r13) : "l"(ADR(13)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r14) : "l"(ADR(14)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r15) : "l"(ADR(15)), "r"(1u));
        v0 ^= r0^r1^r2^r3; v4 ^= r4^r5^r6^r7;
        v8 ^= r8^r9^r10^r11; v12 ^= r12^r13^r14^r15;
#elif MODE == 3 || MODE == 4 || MODE == 5 || MODE == 6
        // Per-thread coalesced: UNROLL independent chains
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
        unsigned int r8, r9, r10, r11, r12, r13, r14, r15;
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r0) : "l"(ADR(0)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r1) : "l"(ADR(1)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r2) : "l"(ADR(2)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r3) : "l"(ADR(3)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r4) : "l"(ADR(4)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r5) : "l"(ADR(5)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r6) : "l"(ADR(6)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r7) : "l"(ADR(7)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r8) : "l"(ADR(8)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r9) : "l"(ADR(9)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r10) : "l"(ADR(10)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r11) : "l"(ADR(11)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r12) : "l"(ADR(12)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r13) : "l"(ADR(13)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r14) : "l"(ADR(14)), "r"(1u));
        asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(r15) : "l"(ADR(15)), "r"(1u));
        v0 ^= r0^r1^r2^r3; v4 ^= r4^r5^r6^r7;
        v8 ^= r8^r9^r10^r11; v12 ^= r12^r13^r14^r15;
#elif MODE == 9
        // relaxed.gpu scope
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
        unsigned int r8, r9, r10, r11, r12, r13, r14, r15;
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r0) : "l"(ADR(0)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r1) : "l"(ADR(1)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r2) : "l"(ADR(2)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r3) : "l"(ADR(3)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r4) : "l"(ADR(4)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r5) : "l"(ADR(5)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r6) : "l"(ADR(6)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r7) : "l"(ADR(7)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r8) : "l"(ADR(8)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r9) : "l"(ADR(9)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r10) : "l"(ADR(10)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r11) : "l"(ADR(11)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r12) : "l"(ADR(12)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r13) : "l"(ADR(13)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r14) : "l"(ADR(14)), "r"(1u));
        asm volatile("atom.relaxed.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r15) : "l"(ADR(15)), "r"(1u));
        v0 ^= r0^r1^r2^r3; v4 ^= r4^r5^r6^r7;
        v8 ^= r8^r9^r10^r11; v12 ^= r12^r13^r14^r15;
#elif MODE == 10
        // u64 coalesced stride=8B
        unsigned long long r0, r1, r2, r3, r4, r5, r6, r7;
        unsigned long long r8, r9, r10, r11, r12, r13, r14, r15;
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r0) : "l"(ADR(0)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r1) : "l"(ADR(1)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r2) : "l"(ADR(2)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r3) : "l"(ADR(3)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r4) : "l"(ADR(4)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r5) : "l"(ADR(5)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r6) : "l"(ADR(6)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r7) : "l"(ADR(7)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r8) : "l"(ADR(8)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r9) : "l"(ADR(9)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r10) : "l"(ADR(10)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r11) : "l"(ADR(11)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r12) : "l"(ADR(12)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r13) : "l"(ADR(13)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r14) : "l"(ADR(14)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r15) : "l"(ADR(15)), "l"(1ULL));
        v0 ^= (unsigned)(r0^r1^r2^r3); v4 ^= (unsigned)(r4^r5^r6^r7);
        v8 ^= (unsigned)(r8^r9^r10^r11); v12 ^= (unsigned)(r12^r13^r14^r15);
#elif MODE == 11
        // u64 uncoalesced stride=256B
        unsigned long long r0, r1, r2, r3, r4, r5, r6, r7;
        unsigned long long r8, r9, r10, r11, r12, r13, r14, r15;
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r0) : "l"(ADR(0)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r1) : "l"(ADR(1)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r2) : "l"(ADR(2)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r3) : "l"(ADR(3)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r4) : "l"(ADR(4)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r5) : "l"(ADR(5)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r6) : "l"(ADR(6)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r7) : "l"(ADR(7)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r8) : "l"(ADR(8)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r9) : "l"(ADR(9)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r10) : "l"(ADR(10)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r11) : "l"(ADR(11)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r12) : "l"(ADR(12)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r13) : "l"(ADR(13)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r14) : "l"(ADR(14)), "l"(1ULL));
        asm volatile("atom.global.add.u64 %0, [%1], %2;" : "=l"(r15) : "l"(ADR(15)), "l"(1ULL));
        v0 ^= (unsigned)(r0^r1^r2^r3); v4 ^= (unsigned)(r4^r5^r6^r7);
        v8 ^= (unsigned)(r8^r9^r10^r11); v12 ^= (unsigned)(r12^r13^r14^r15);
#elif MODE == 12
        // atomicCAS u32 coalesced stride=4B
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
        unsigned int r8, r9, r10, r11, r12, r13, r14, r15;
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r0) : "l"(ADR(0)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r1) : "l"(ADR(1)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r2) : "l"(ADR(2)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r3) : "l"(ADR(3)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r4) : "l"(ADR(4)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r5) : "l"(ADR(5)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r6) : "l"(ADR(6)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r7) : "l"(ADR(7)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r8) : "l"(ADR(8)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r9) : "l"(ADR(9)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r10) : "l"(ADR(10)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r11) : "l"(ADR(11)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r12) : "l"(ADR(12)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r13) : "l"(ADR(13)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r14) : "l"(ADR(14)), "r"(0u), "r"(1u));
        asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(r15) : "l"(ADR(15)), "r"(0u), "r"(1u));
        v0 ^= r0^r1^r2^r3; v4 ^= r4^r5^r6^r7;
        v8 ^= r8^r9^r10^r11; v12 ^= r12^r13^r14^r15;
#elif MODE == 13
        // red.global (no return value)
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(0)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(1)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(2)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(3)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(4)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(5)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(6)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(7)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(8)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(9)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(10)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(11)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(12)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(13)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(14)), "r"(1u));
        asm volatile("red.global.add.u32 [%0], %1;" :: "l"(ADR(15)), "r"(1u));
        // Keep v0 live
        v0 ^= (unsigned)i;
#elif MODE == 14
        // atom.acquire.gpu — acquire ordering on return
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
        unsigned int r8, r9, r10, r11, r12, r13, r14, r15;
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r0) : "l"(ADR(0)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r1) : "l"(ADR(1)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r2) : "l"(ADR(2)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r3) : "l"(ADR(3)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r4) : "l"(ADR(4)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r5) : "l"(ADR(5)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r6) : "l"(ADR(6)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r7) : "l"(ADR(7)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r8) : "l"(ADR(8)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r9) : "l"(ADR(9)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r10) : "l"(ADR(10)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r11) : "l"(ADR(11)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r12) : "l"(ADR(12)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r13) : "l"(ADR(13)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r14) : "l"(ADR(14)), "r"(1u));
        asm volatile("atom.acquire.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r15) : "l"(ADR(15)), "r"(1u));
        v0 ^= r0^r1^r2^r3; v4 ^= r4^r5^r6^r7;
        v8 ^= r8^r9^r10^r11; v12 ^= r12^r13^r14^r15;
#elif MODE == 15
        // atom.release.gpu — release ordering on write
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
        unsigned int r8, r9, r10, r11, r12, r13, r14, r15;
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r0) : "l"(ADR(0)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r1) : "l"(ADR(1)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r2) : "l"(ADR(2)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r3) : "l"(ADR(3)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r4) : "l"(ADR(4)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r5) : "l"(ADR(5)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r6) : "l"(ADR(6)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r7) : "l"(ADR(7)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r8) : "l"(ADR(8)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r9) : "l"(ADR(9)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r10) : "l"(ADR(10)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r11) : "l"(ADR(11)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r12) : "l"(ADR(12)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r13) : "l"(ADR(13)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r14) : "l"(ADR(14)), "r"(1u));
        asm volatile("atom.release.gpu.global.add.u32 %0, [%1], %2;" : "=r"(r15) : "l"(ADR(15)), "r"(1u));
        v0 ^= r0^r1^r2^r3; v4 ^= r4^r5^r6^r7;
        v8 ^= r8^r9^r10^r11; v12 ^= r12^r13^r14^r15;
#endif
    }

    // Unconditional use to defeat DCE
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7^v8^v9^v10^v11^v12^v13^v14^v15;
    if ((int)acc == seed) ((unsigned int*)C)[gtid] = acc;
}
