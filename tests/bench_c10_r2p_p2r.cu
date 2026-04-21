// C10: R2P (register→predicate) and P2R (predicate→register) cost
// Modes:
// 0: setp.ne.b32 (R2P style — comparison turning reg into predicate)
// 1: selp.b32 (P2R style — predicate selecting register value)
// 2: PTX explicit P2R (selp .b32)
// 3: round-trip R2P + P2R chain
// 4: baseline IADD3 (Cluster B)
// 5: baseline FFMA (Cluster A)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int x = 0xDEADBEEFu ^ (unsigned)u2;
    unsigned int y = 0xCAFEBABEu;
    float fa = (float)threadIdx.x * 0.001f + (float)u2 * 1e-9f;
    float fb = 1.0001f, fc = 0.5f;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // Pure setp (R2P-style): chained — predicate consumed, fed back as int
            asm volatile("{ .reg .pred p;\n"
                         "  setp.ne.b32 p, %0, %1;\n"
                         "  selp.b32 %0, %1, %2, p; }"
                         : "+r"(v) : "r"(x), "r"(y));
#elif MODE == 1
            // Pure selp (P2R) — uses pre-computed predicate
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %1, 0;\n"
                         "  selp.b32 %0, %1, %2, p; }"
                         : "+r"(v) : "r"(x), "r"(y));
#elif MODE == 2
            // R2P + P2R chained: setp from v, then selp using that predicate, twice
            asm volatile("{ .reg .pred p;\n"
                         "  setp.ne.b32 p, %0, %1;\n"
                         "  selp.b32 %0, %1, %2, p;\n"
                         "  setp.eq.b32 p, %0, 0;\n"
                         "  selp.b32 %0, %2, %1, p; }"
                         : "+r"(v) : "r"(x), "r"(y));
#elif MODE == 3
            // BFI (bit field insert) — uses predicate-style internal logic, baseline
            asm("bfi.b32 %0, %0, %1, 4, 8;" : "+r"(v) : "r"(x));
#elif MODE == 4
            // IADD3 baseline (Cluster B, ~6 cy chained)
            asm("add.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
#elif MODE == 5
            // FFMA baseline (Cluster A, ~6 cy chained)
            fa = fa * fb + fc;
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed && (int)fa == seed) C[blockIdx.x] = (float)v + fa;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f cy/op=%.4f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/16.0);
    }
}
