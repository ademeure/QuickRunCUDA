// Isolate per-read cost by CHAIN depth (no consume) and by read-only with pure adder.
// UNROLL = reads per iter.
// OP=0: %clock64 (CS2R SR_CLOCKLO)
// OP=1: %clock  (u32, S2UR expected)
// OP=2: %clock64 back-to-back INDEPENDENT (DCE-proof via xor-chain into acc)
// OP=3: %clock  u32 back-to-back independent
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(32, 1) void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
    unsigned long long acc = seed;
    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned long long v;
#if OP == 0
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(v));
            acc ^= v;
#elif OP == 1
            unsigned x;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(x));
            acc ^= x;
#elif OP == 2
            // independent reads (no inter-read dep, XOR into acc at end)
            asm volatile("mov.u64 %0, %%clock64;" : "=l"(v));
            acc += v + i + j;
#elif OP == 3
            unsigned x;
            asm volatile("mov.u32 %0, %%clock;" : "=r"(x));
            acc += x + i + j;
#endif
        }
    }
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
    if (threadIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
        ((unsigned long long*)C)[1] = acc;
    }
}
