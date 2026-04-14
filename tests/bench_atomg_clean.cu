// Global-memory atomics, per-thread unique addresses (no cacheline contention).

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v0=0,v1=0,v2=0,v3=0,v4=0,v5=0,v6=0,v7=0;
    // 128-byte stride per lane across chains, so warps don't cacheline-conflict
    unsigned long long tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned long long base = (unsigned long long)A + tid * 32;  // each thread owns 32B

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
#define G(k) (base + (k) * 4ULL)
#if OP == 0 // atom.global.add.u32
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v0) : "l"(G(0)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v1) : "l"(G(1)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v2) : "l"(G(2)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v3) : "l"(G(3)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v4) : "l"(G(4)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v5) : "l"(G(5)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v6) : "l"(G(6)), "r"(1u));
            asm volatile("atom.global.add.u32 %0, [%1], %2;" : "=r"(v7) : "l"(G(7)), "r"(1u));
#elif OP == 1 // atom.global.min.u32
            asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(v0) : "l"(G(0)), "r"(0xFu));
            asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(v1) : "l"(G(1)), "r"(0xFu));
            asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(v2) : "l"(G(2)), "r"(0xFu));
            asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(v3) : "l"(G(3)), "r"(0xFu));
            asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(v4) : "l"(G(4)), "r"(0xFu));
            asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(v5) : "l"(G(5)), "r"(0xFu));
            asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(v6) : "l"(G(6)), "r"(0xFu));
            asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(v7) : "l"(G(7)), "r"(0xFu));
#elif OP == 2 // atom.global.cas.b32
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(v0) : "l"(G(0)), "r"(0u), "r"(1u));
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(v1) : "l"(G(1)), "r"(0u), "r"(1u));
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(v2) : "l"(G(2)), "r"(0u), "r"(1u));
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(v3) : "l"(G(3)), "r"(0u), "r"(1u));
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(v4) : "l"(G(4)), "r"(0u), "r"(1u));
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(v5) : "l"(G(5)), "r"(0u), "r"(1u));
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(v6) : "l"(G(6)), "r"(0u), "r"(1u));
            asm volatile("atom.global.cas.b32 %0, [%1], %2, %3;" : "=r"(v7) : "l"(G(7)), "r"(0u), "r"(1u));
#elif OP == 3 // atom.global.exch
            asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(v0) : "l"(G(0)), "r"(1u));
            asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(v1) : "l"(G(1)), "r"(1u));
            asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(v2) : "l"(G(2)), "r"(1u));
            asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(v3) : "l"(G(3)), "r"(1u));
            asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(v4) : "l"(G(4)), "r"(1u));
            asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(v5) : "l"(G(5)), "r"(1u));
            asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(v6) : "l"(G(6)), "r"(1u));
            asm volatile("atom.global.exch.b32 %0, [%1], %2;" : "=r"(v7) : "l"(G(7)), "r"(1u));
#elif OP == 4 // red.global.add.u32 (no return)
            asm volatile("red.global.add.u32 [%0], %1;" :: "l"(G(0)), "r"(1u));
            asm volatile("red.global.add.u32 [%0], %1;" :: "l"(G(1)), "r"(1u));
            asm volatile("red.global.add.u32 [%0], %1;" :: "l"(G(2)), "r"(1u));
            asm volatile("red.global.add.u32 [%0], %1;" :: "l"(G(3)), "r"(1u));
            asm volatile("red.global.add.u32 [%0], %1;" :: "l"(G(4)), "r"(1u));
            asm volatile("red.global.add.u32 [%0], %1;" :: "l"(G(5)), "r"(1u));
            asm volatile("red.global.add.u32 [%0], %1;" :: "l"(G(6)), "r"(1u));
            asm volatile("red.global.add.u32 [%0], %1;" :: "l"(G(7)), "r"(1u));
#elif OP == 5 // atom.global.add.f32
            { float f0=0,f1=0,f2=0,f3=0,f4=0,f5=0,f6=0,f7=0;
              asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(f0) : "l"(G(0)), "f"(1.0f));
              asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(f1) : "l"(G(1)), "f"(1.0f));
              asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(f2) : "l"(G(2)), "f"(1.0f));
              asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(f3) : "l"(G(3)), "f"(1.0f));
              asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(f4) : "l"(G(4)), "f"(1.0f));
              asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(f5) : "l"(G(5)), "f"(1.0f));
              asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(f6) : "l"(G(6)), "f"(1.0f));
              asm volatile("atom.global.add.f32 %0, [%1], %2;" : "=f"(f7) : "l"(G(7)), "f"(1.0f));
              v0=__float_as_int(f0);v1=__float_as_int(f1);v2=__float_as_int(f2);v3=__float_as_int(f3);
              v4=__float_as_int(f4);v5=__float_as_int(f5);v6=__float_as_int(f6);v7=__float_as_int(f7); }
#endif
        }
    }
    unsigned int acc = v0^v1^v2^v3^v4^v5^v6^v7;
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
