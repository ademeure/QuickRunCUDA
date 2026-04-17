// Smem atomics audit: all variants of atom.shared.

#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef ITERS
#define ITERS 1024
#endif
#ifndef OP
#define OP 0
#endif

extern __shared__ unsigned smem[];

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(unsigned* A, unsigned* B, unsigned* C, int u0, int seed, int u2) {
    if (threadIdx.x == 0) for (int i = 0; i < 32; i++) smem[i] = i;
    __syncthreads();
    unsigned v = (unsigned)seed + threadIdx.x;
    unsigned base = (unsigned)__cvta_generic_to_shared(&smem[0]);
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        unsigned r;
#if OP == 0
        asm volatile("atom.shared.add.u32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(v));
#elif OP == 1
        asm volatile("atom.shared.cas.b32 %0, [%1], %2, %3;" : "=r"(r) : "r"(base), "r"(v), "r"(v+1));
#elif OP == 2
        asm volatile("atom.shared.exch.b32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(v));
#elif OP == 3
        asm volatile("atom.shared.min.u32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(v));
#elif OP == 4
        asm volatile("atom.shared.max.u32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(v));
#elif OP == 5
        asm volatile("atom.shared.inc.u32 %0, [%1], 0xffffffff;" : "=r"(r) : "r"(base));
#elif OP == 6
        asm volatile("atom.shared.dec.u32 %0, [%1], 0xffffffff;" : "=r"(r) : "r"(base));
#elif OP == 7
        asm volatile("atom.shared.and.b32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(v));
#elif OP == 8
        asm volatile("atom.shared.or.b32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(v));
#elif OP == 9
        asm volatile("atom.shared.xor.b32 %0, [%1], %2;" : "=r"(r) : "r"(base), "r"(v));
#elif OP == 10
        // FP32 add (no native, expect emulation)
        float fv = (float)v;
        float fr;
        asm volatile("atom.shared.add.f32 %0, [%1], %2;" : "=f"(fr) : "r"(base), "f"(fv));
        r = __float_as_int(fr);
#elif OP == 11
        // red.shared (no return)
        asm volatile("red.shared.add.u32 [%0], %1;" :: "r"(base), "r"(v) : "memory");
        r = 0;
#endif
        v = r + 1;
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        C[2] = v;
    }
}
