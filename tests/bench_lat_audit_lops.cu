// Audit: LOP3 / SHF latency via single-thread serial chain
#ifndef CHAIN_LEN
#define CHAIN_LEN 4096
#endif
#ifndef OP
#define OP 0  // 0=LOP3 1=SHF.L 2=IMAD.HI 3=HFMA2 4=HADD2
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(int* A, int* B, int* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0) return;

    unsigned a = (unsigned)(seed + 1);
    unsigned b = 0xDEADBEEFu;
#if OP == 3 || OP == 4
    // half2 packed
    unsigned ha = 0x3C003C00u;  // {1.0h, 1.0h}
    unsigned hb = 0x3800B000u;  // {0.5h, -0.5h}
#endif

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 32
    for (int i = 0; i < CHAIN_LEN; i++) {
#if OP == 0
        // LOP3: a = (a & b) ^ b
        asm volatile("lop3.b32 %0, %0, %1, %1, 0x96;" : "+r"(a) : "r"(b));
#elif OP == 1
        // SHF.L: a = funnel-shift left
        asm volatile("shf.l.wrap.b32 %0, %0, %1, 1;" : "+r"(a) : "r"(b));
#elif OP == 2
        // IMAD.HI.U32: only upper 32 bits of a*b + a
        asm volatile("mad.hi.u32 %0, %0, %1, %0;" : "+r"(a) : "r"(b));
#elif OP == 3
        // HFMA2: half2 fma
        asm volatile("fma.rn.f16x2 %0, %0, %1, %0;" : "+r"(ha) : "r"(hb));
#elif OP == 4
        // HADD2: half2 add
        asm volatile("add.f16x2 %0, %0, %1;" : "+r"(ha) : "r"(hb));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (blockIdx.x == 0) {
        ((unsigned long long*)C)[0] = t1 - t0;
#if OP == 3 || OP == 4
        ((unsigned*)C)[2] = ha;
#else
        ((unsigned*)C)[2] = a;
#endif
    }
}
