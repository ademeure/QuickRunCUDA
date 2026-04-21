// M8: Carry propagation in IADD chains
// Test: 64-bit add via add.cc + addc vs IADD3 (no carry chain)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v_lo = (unsigned)(threadIdx.x ^ u2);
    unsigned int v_hi = (unsigned)(threadIdx.x + u2);
    unsigned int x_lo = 0xDEADBEEFu ^ (unsigned)u2;
    unsigned int x_hi = 0xCAFEBABEu;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
#if MODE == 0
            // Native 64-bit add via add.cc + addc
            asm volatile("{ add.cc.u32 %0, %0, %2;\n"
                         "  addc.u32   %1, %1, %3; }"
                         : "+r"(v_lo), "+r"(v_hi) : "r"(x_lo), "r"(x_hi));
#elif MODE == 1
            // IADD3 (no carry — independent ops)
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v_lo) : "r"(x_lo));
            asm volatile("add.u32 %0, %0, %1;" : "+r"(v_hi) : "r"(x_hi));
#elif MODE == 2
            // 2 independent IADD3 without carry chain (just for comparison)
            asm volatile("add.u32 %0, %0, %2;\n"
                         "add.u32 %1, %1, %3;"
                         : "+r"(v_lo), "+r"(v_hi) : "r"(x_lo), "r"(x_hi));
#elif MODE == 3
            // Manual carry via setp + selp
            asm volatile("{ .reg .pred c;\n"
                         "  add.cc.u32 %0, %0, %2;\n"
                         "  add.u32 %1, %1, %3;\n"
                         "  setp.lo.u32 c, %0, %2;\n"
                         "  @c add.u32 %1, %1, 1; }"
                         : "+r"(v_lo), "+r"(v_hi) : "r"(x_lo), "r"(x_hi));
#elif MODE == 4
            // Native u64 add via add.u64 (compiler-emitted)
            unsigned long long v64 = ((unsigned long long)v_hi << 32) | v_lo;
            unsigned long long x64 = ((unsigned long long)x_hi << 32) | x_lo;
            asm volatile("add.u64 %0, %0, %1;" : "+l"(v64) : "l"(x64));
            v_lo = (unsigned)v64;
            v_hi = (unsigned)(v64 >> 32);
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v_lo == (unsigned)seed && v_hi == (unsigned)seed) C[blockIdx.x] = (float)v_lo;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f cy/op=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS/16.0);
    }
}
