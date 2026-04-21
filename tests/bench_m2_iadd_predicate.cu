// M2: IADD3 + predicate cost — branchless code patterns
// MODE 0: pure IADD3 baseline
// MODE 1: @p IADD3 (predicated, always true uniform)
// MODE 2: @p IADD3 (predicated, always false uniform)
// MODE 3: @p IADD3 + @!p IADD3 (both arms emitted; like ?:)
// MODE 4: 4 IADD3 baseline
// MODE 5: 4 @p IADD3 (predicated)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int x = 0xDEADBEEFu ^ (unsigned)u2;

    // Runtime predicate
    unsigned int p_in;
#if MODE == 1 || MODE == 5
    p_in = (u2 != -999) ? 1 : 0;  // always true
#elif MODE == 2
    p_in = (u2 == -999) ? 1 : 0;  // always false
#elif MODE == 3
    p_in = ((unsigned)threadIdx.x ^ (unsigned)u2) & 1;  // half-warp split
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v) : "r"(x));
#elif MODE == 1 || MODE == 2
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n"
                         "  @p add.u32 %0, %0, %1; }"
                         : "+r"(v) : "r"(x), "r"(p_in));
#elif MODE == 3
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n"
                         "  @p add.u32 %0, %0, %1;\n"
                         "  @!p add.u32 %0, %0, %1; }"
                         : "+r"(v) : "r"(x), "r"(p_in));
#elif MODE == 4
            asm volatile("add.u32 %0, %0, %1;\n"
                         "add.u32 %0, %0, %1;\n"
                         "add.u32 %0, %0, %1;\n"
                         "add.u32 %0, %0, %1;"
                         : "+r"(v) : "r"(x));
#elif MODE == 5
            asm volatile("{ .reg .pred p; setp.ne.b32 p, %2, 0;\n"
                         "  @p add.u32 %0, %0, %1;\n"
                         "  @p add.u32 %0, %0, %1;\n"
                         "  @p add.u32 %0, %0, %1;\n"
                         "  @p add.u32 %0, %0, %1; }"
                         : "+r"(v) : "r"(x), "r"(p_in));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/op=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS/16.0);
    }
}
