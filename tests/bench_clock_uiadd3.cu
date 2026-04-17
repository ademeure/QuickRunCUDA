// Force compiler to use UIADD3 (uniform sub) for clock diff.
// Trick: keep both clocks in URegs by feeding them only into uniform-pipe ops.
// Then accumulate uniformly, mov to vector at end, store.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    unsigned lane;
    asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
#if OP == 0
    // 2 clocks, try keeping uniform via XOR + uniform output
    if (lane == 0) {
        unsigned t0, t1;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        unsigned diff = t1 - t0;
        C[blockIdx.x] = diff;
    }
#elif OP == 1
    // Force 2 S2URs by using both as predicate inputs (uniform use)
    if (lane == 0) {
        unsigned t0, t1;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        // Uniform XOR — should stay in URegs
        unsigned diff = t0 ^ t1;
        C[blockIdx.x] = diff;
    }
#elif OP == 2
    // 10 clocks accumulated — many uniform values, see if compiler keeps URegs
    if (lane == 0) {
        unsigned acc = 0;
        #pragma unroll
        for (int k = 0; k < 10; k++) {
            unsigned t;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
            acc += t;
        }
        C[blockIdx.x] = acc;
    }
#elif OP == 3
    // 10 clocks with FFMA between — see if compiler optimizes pipe usage
    float v0 = __int_as_float(blockIdx.x+1)*1e-30f;
    float v1 = __int_as_float(blockIdx.x+2)*1e-30f;
    float y = 1.5f;
    if (lane == 0) {
        unsigned acc = 0;
        #pragma unroll
        for (int k = 0; k < 10; k++) {
            unsigned t;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(t));
            acc += t;
            v0 = v0*y + v0;
            v1 = v1*y + v1;
        }
        C[blockIdx.x] = acc;
        if (__float_as_int(v0+v1) == seed) C[blockIdx.x+1024] = v0+v1;
    }
#elif OP == 4
    // Inline asm: explicit UIADD3 hint via PTX add.cc.u32 chain
    if (lane == 0) {
        unsigned t0, t1, diff;
        asm volatile(
            "{ .reg .b32 %lo;"
            "  mov.u32 %lo, %%clock; "
            "  mov.u32 %0, %%clock; "
            "  sub.u32 %1, %0, %lo; }"
            : "=r"(t1), "=r"(diff));
        C[blockIdx.x] = diff;
    }
#elif OP == 5
    // Try storing via atomic exchange — does that allow uniform reg input?
    if (lane == 0) {
        unsigned t0, t1;
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t0));
        asm volatile("mov.u32 %0, %%clock;" : "=r"(t1));
        unsigned diff = t1 - t0;
        unsigned old;
        asm volatile("atom.global.exch.b32 %0, [%1], %2;"
            : "=r"(old) : "l"(C + blockIdx.x), "r"(diff) : "memory");
    }
#endif
}
