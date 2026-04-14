// Integer op latencies — clock64 bracketed, chain.

#ifndef N_OPS
#define N_OPS 128
#endif
#ifndef ITERS_OUTER
#define ITERS_OUTER 256
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    unsigned int x = 0x12345678u;
    unsigned int y = 0xABCDEF01u;
    unsigned long long l = 0x123456789ABCDEF0ULL;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0  // IMAD (mul-add chain)
            asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(x));
#elif OP == 1 // IADD3 via inline runtime
            asm volatile("add.u32 %0, %0, 7;" : "+r"(x));
#elif OP == 2 // IADD3 ternary
            asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(y));
#elif OP == 3 // IMUL.HI
            asm volatile("mul.hi.u32 %0, %0, %1;" : "+r"(x) : "r"(0x9E3779B9u));
#elif OP == 4 // LOP3 XOR with varying const
            asm volatile("xor.b32 %0, %0, 0xDEADBEEF;" : "+r"(x));
#elif OP == 5 // LOP3 XOR via register
            asm volatile("xor.b32 %0, %0, %1;" : "+r"(x) : "r"(y));
#elif OP == 6 // SHF.L funnel shift
            asm volatile("shf.l.wrap.b32 %0, %0, %0, 5;" : "+r"(x));
#elif OP == 7 // SHL by 1
            asm volatile("shl.b32 %0, %0, 1;" : "+r"(x));
#elif OP == 8 // PRMT with register control
            asm volatile("prmt.b32 %0, %0, %0, %1;" : "+r"(x) : "r"(y & 0xFFFFu));
#elif OP == 9 // ISETP + SEL
            unsigned int r;
            asm volatile("{.reg .pred p; setp.lt.u32 p, %1, 0x80000000; selp.u32 %0, %1, %2, p;}" : "=r"(r) : "r"(x), "r"(y));
            x = r;
#elif OP == 10 // u64 ADD
            asm volatile("add.u64 %0, %0, %1;" : "+l"(l) : "l"((unsigned long long)x));
            x = (unsigned int)l;
#elif OP == 11 // u64 MUL.LO
            asm volatile("mul.lo.u64 %0, %0, %1;" : "+l"(l) : "l"(0xDEADBEEFULL));
            x = (unsigned int)l;
#elif OP == 12 // LEA pattern
            asm volatile("mad.wide.u32 %0, %1, 4, %0;" : "+l"(l) : "r"(x));
            x = (unsigned int)l;
#elif OP == 13 // POPC
            unsigned int r;
            asm volatile("popc.b32 %0, %1;" : "=r"(r) : "r"(x));
            x = r + y;
#elif OP == 14 // BREV
            asm volatile("brev.b32 %0, %0;" : "+r"(x));
#elif OP == 15 // FLO (find leading one)
            unsigned int r;
            asm volatile("bfind.u32 %0, %1;" : "=r"(r) : "r"(x | 1));
            x = x + r;
#endif
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if ((int)x == seed || l == (unsigned long long)seed) ((unsigned int*)C)[0] = x + (unsigned)l;
    ((unsigned long long*)C)[1] = total_dt;
}
