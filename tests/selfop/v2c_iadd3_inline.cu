// Test 2c-prime: force IADD3 inline (no compiler choice between IMAD/IADD3)
// PTX add.u32 d,a,a -- self-op IADD via inline; force ASM volatile
// We use a true 3-source IADD3 via PTX `add.u32` doesn't accept 3 src
// so use 2 separate adds.  Or let's use sub-and-add via `mad.lo` for parity.
// Simpler: explicit ptxas pragma not possible; verify by SASS.
#ifndef N_INNER
#define N_INNER 1024
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int seed, int u1, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned int)threadIdx.x + 1u;
    unsigned int k1 = (unsigned int)u1;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    // To force IADD3 self-op pattern at SASS level, use chain-via-different-reg
    // but keep ALL other operands = same reg.  This breaks the "all same"
    // illegal-decode constraint that pushed compiler to IMAD.IADD.
    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
            // d = d + d + k1   (3-source) -- compiler should emit IADD3 R,R,R,k1
            asm volatile("add.u32 %0, %0, %0; add.u32 %0, %0, %1;" : "+r"(v) : "r"(k1));
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if ((int)v == seed) C[blockIdx.x] = (float)v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        // 2 adds per inner iter
        unsigned long long total = (unsigned long long)N_OUTER * N_INNER * 2ULL;
        printf("v2c_iadd3_inline total=%llu clk=%llu cy/op=%.4f\n",
               total, t1 - t0, (double)(t1-t0)/(double)total);
    }
}
