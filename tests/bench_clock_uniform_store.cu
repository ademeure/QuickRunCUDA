// Investigate uniform-register clock diff + direct uniform store to DRAM.
// Goal: ZERO vector register usage for timing — everything in uniform pipe.
//
// OP=0 : naive — clock as u32 in vector regs, sub, store
// OP=1 : explicit u32 clock + ULOP3-style logic
// OP=2 : try to force uniform: mov.u32 dst, %clock; bare sub; st via "r" constraint
// OP=3 : __noinline__ approach + uniform asm
// OP=4 : warp-uniform — only thread 0 stores

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
#if OP == 0
    // Baseline: read clock, do something, read clock, subtract, store
    unsigned t0, t1;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
    // Some no-op work
    unsigned x = seed * threadIdx.x;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
    if (threadIdx.x == 0) C[blockIdx.x] = t1 - t0;
    if (threadIdx.x == 0) C[blockIdx.x + 1024] = x;  // prevent dead code
#elif OP == 1
    // Try u64 clock64, take diff in u64, write low half
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
#elif OP == 2
    // Capture, store ONLY in thread 0 — see if compiler uses uniform regs for the diff
    if (threadIdx.x == 0) {
        unsigned t0, t1;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        C[blockIdx.x] = t1 - t0;
    }
#elif OP == 3
    // u64 in thread 0 only
    if (threadIdx.x == 0) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
    }
#elif OP == 4
    // Use bar.warp.sync to ensure WHOLE-WARP uniform behavior
    unsigned t0, t1;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
    __syncwarp();
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
    __syncwarp();
    if (threadIdx.x == 0) C[blockIdx.x] = t1 - t0;
#elif OP == 5
    // Try using ULDG/USTG-style — explicit uniform store via UMOV→stmem
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        unsigned long long diff = t1 - t0;
        // Hint to compiler: use uniform store
        asm volatile("st.global.u64 [%0], %1;" :: "l"((unsigned long long*)C), "l"(diff) : "memory");
    }
#elif OP == 6
    // Many clock samples in thread 0 — see if compiler uses uniform pipe
    if (threadIdx.x == 0) {
        unsigned long long t[8];
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[0]));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[1]));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[2]));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[3]));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[4]));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[5]));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[6]));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t[7]));
        unsigned long long *out = (unsigned long long*)C + blockIdx.x * 8;
        out[0] = t[1]-t[0]; out[1] = t[2]-t[1]; out[2] = t[3]-t[2]; out[3] = t[4]-t[3];
        out[4] = t[5]-t[4]; out[5] = t[6]-t[5]; out[6] = t[7]-t[6]; out[7] = t[7]-t[0];
    }
#elif OP == 7
    // Force PTX-level local regs that compiler should map to uniform
    if (threadIdx.x == 0) {
        unsigned d;
        asm volatile(
            "{ .reg .b32 r1, r2; "
            "  mov.u32 r1, %%clock; "
            "  mov.u32 r2, %%clock; "
            "  sub.u32 %0, r2, r1; }"
            : "=r"(d));
        C[blockIdx.x] = d;
    }
#elif OP == 8
    // Same with u64
    if (threadIdx.x == 0) {
        unsigned long long d;
        asm volatile(
            "{ .reg .b64 r1, r2; "
            "  mov.u64 r1, %%clock64; "
            "  mov.u64 r2, %%clock64; "
            "  sub.u64 %0, r2, r1; }"
            : "=l"(d));
        ((unsigned long long*)C)[blockIdx.x] = d;
    }
#elif OP == 9
    // Try store FROM URegs explicitly with __builtin_assume_aligned tricks
    if (threadIdx.x == 0) {
        unsigned t0, t1;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        unsigned diff = t1 - t0;
        asm volatile("st.global.u32 [%0], %1;" :: "l"(C + blockIdx.x), "r"(diff) : "memory");
    }
#elif OP == 10
    // Use only thread 0 of WHOLE GRID — minimal launches
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        unsigned long long diff = t1 - t0;
        asm volatile("st.global.u64 [%0], %1;" :: "l"(C), "l"(diff) : "memory");
    }
#elif OP == 11
    // Try forcing FULL uniform path: 2 S2UR + 1 UIADD3 (uniform sub) + 1 store
    if (threadIdx.x == 0) {
        unsigned t0, t1;
        // Force u32 reads (S2UR routing) by NOT using a u64 clock
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        // Subtract as u32 only — should stay in U pipe if compiler keeps both URegs
        unsigned diff = t1 - t0;
        // Store: cast diff explicitly so compiler knows it's a u32 store
        C[blockIdx.x] = diff;
    }
#elif OP == 12
    // Like OP=11 but with the address load also from constant — minimal vector regs
    if (threadIdx.x == 0 && blockIdx.x < 256) {
        unsigned t0, t1;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        // Try to keep both clocks as uniform — explicit asm sub
        unsigned diff;
        asm volatile("sub.u32 %0, %1, %2;" : "=r"(diff) : "r"(t1), "r"(t0));
        C[blockIdx.x] = diff;
    }
#elif OP == 13
    // Use laneid instead of threadIdx.x for compiler clarity
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    if (lane == 0) {
        unsigned t0, t1;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        C[blockIdx.x] = t1 - t0;
    }
#elif OP == 14
    // laneid + u64 clock64
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    if (lane == 0) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        ((unsigned long long*)C)[blockIdx.x] = t1 - t0;
    }
#elif OP == 15
    // ALL lanes participate (whole warp), only lane 0 stores
    unsigned t0, t1;
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
    asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
    if (lane == 0) {
        C[blockIdx.x] = t1 - t0;
    }
#endif
}
