// Chain latency: pure IMAD chain vs IMAD+LOP3 alternating chain.
// Single warp, dependent chain through one register.

#ifndef N_INNER
#define N_INNER 1000
#endif
#ifndef N_OUTER
#define N_OUTER 100
#endif
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)threadIdx.x + 1u;
    unsigned int b = 0xDEADBEEFu;
    unsigned int c = 0xC0FFEE00u;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < N_OUTER; i++) {
        #pragma unroll N_INNER
        for (int j = 0; j < N_INNER; j++) {
#if MODE == 0
            // Pure IMAD chain (mad.lo writes back to v, reads v + b + c)
            asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(b), "r"(c));
#elif MODE == 1
            // Pure LOP3 chain
            asm volatile("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v) : "r"(b), "r"(c));
#elif MODE == 2
            // IMAD -> LOP3 alternating (chain is in v)
            asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(b), "r"(c));
            asm volatile("lop3.b32  %0, %0, %1, %2, 0x96;" : "+r"(v) : "r"(b), "r"(c));
#elif MODE == 3
            // IMAD -> IADD chain
            asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(b), "r"(c));
            asm volatile("add.u32    %0, %0, %1;"     : "+r"(v) : "r"(b));
#elif MODE == 4
            // FFMA -> IMAD alternating (chain via cast)
            float fv = __uint_as_float(v);
            asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(__uint_as_float(b)), "f"(__uint_as_float(c)));
            v = __float_as_uint(fv);
            asm volatile("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(b), "r"(c));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        unsigned long long total = (unsigned long long)N_OUTER * (unsigned long long)N_INNER;
#if MODE == 2 || MODE == 3 || MODE == 4
        unsigned long long inst = total * 2;  // 2 inst per chain step
#else
        unsigned long long inst = total;
#endif
        printf("MODE=%d N_INNER=%d N_OUTER=%d insts=%llu clk=%llu cy/inst=%.4f cy/chain_step=%.4f\n",
               MODE, N_INNER, N_OUTER, inst, t1 - t0,
               (double)(t1-t0)/(double)inst,
               (double)(t1-t0)/(double)total);
    }
}
