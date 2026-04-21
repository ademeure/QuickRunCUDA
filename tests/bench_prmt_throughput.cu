// PRMT byte-permute throughput vs LOP3, IADD3, IMAD baselines
// PRMT: pick 4 bytes from 8 (2 source registers) using a 16-bit selector
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int s1 = 0xDEADBEEFu ^ (unsigned)u2;
    unsigned int s2 = 0xCAFEBABEu ^ (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        // 4 chained ops per iter
#if MODE == 0
        // PRMT default mode (just byte selection)
        asm("prmt.b32 %0, %0, %1, 0x7531;" : "+r"(v) : "r"(s1));
        asm("prmt.b32 %0, %0, %1, 0x6420;" : "+r"(v) : "r"(s2));
        asm("prmt.b32 %0, %0, %1, 0x3210;" : "+r"(v) : "r"(s1));
        asm("prmt.b32 %0, %0, %1, 0x4567;" : "+r"(v) : "r"(s2));
#elif MODE == 1
        // PRMT.f4e (forward 4 extract — for sign-extending byte→nibble)
        asm("prmt.b32.f4e %0, %0, %1, 0x7531;" : "+r"(v) : "r"(s1));
        asm("prmt.b32.f4e %0, %0, %1, 0x6420;" : "+r"(v) : "r"(s2));
        asm("prmt.b32.f4e %0, %0, %1, 0x3210;" : "+r"(v) : "r"(s1));
        asm("prmt.b32.f4e %0, %0, %1, 0x4567;" : "+r"(v) : "r"(s2));
#elif MODE == 2
        // LOP3 (Cluster B baseline)
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(v) : "r"(s1), "r"(s2));
        asm("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v) : "r"(s1), "r"(s2));
        asm("lop3.b32 %0, %0, %1, %2, 0x6A;" : "+r"(v) : "r"(s1), "r"(s2));
        asm("lop3.b32 %0, %0, %1, %2, 0x55;" : "+r"(v) : "r"(s1), "r"(s2));
#elif MODE == 3
        // IADD3 (Cluster B baseline)
        asm("add.u32 %0, %0, %1;" : "+r"(v) : "r"(s1));
        asm("add.u32 %0, %0, %1;" : "+r"(v) : "r"(s2));
        asm("add.u32 %0, %0, %1;" : "+r"(v) : "r"(s1));
        asm("add.u32 %0, %0, %1;" : "+r"(v) : "r"(s2));
#elif MODE == 4
        // SHF.L (logical shift left — also Cluster B?)
        asm("shf.l.wrap.b32 %0, %0, %1, 4;" : "+r"(v) : "r"(s1));
        asm("shf.l.wrap.b32 %0, %0, %1, 8;" : "+r"(v) : "r"(s2));
        asm("shf.l.wrap.b32 %0, %0, %1, 12;" : "+r"(v) : "r"(s1));
        asm("shf.l.wrap.b32 %0, %0, %1, 16;" : "+r"(v) : "r"(s2));
#elif MODE == 5
        // PRMT alternating with LOP3 — mixed cluster B?
        asm("prmt.b32 %0, %0, %1, 0x7531;" : "+r"(v) : "r"(s1));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(v) : "r"(s1), "r"(s2));
        asm("prmt.b32 %0, %0, %1, 0x6420;" : "+r"(v) : "r"(s2));
        asm("lop3.b32 %0, %0, %1, %2, 0x96;" : "+r"(v) : "r"(s1), "r"(s2));
#elif MODE == 6
        // PRMT alternating with IMAD (cluster A)
        asm("prmt.b32 %0, %0, %1, 0x7531;" : "+r"(v) : "r"(s1));
        asm("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(s1), "r"(s2));
        asm("prmt.b32 %0, %0, %1, 0x6420;" : "+r"(v) : "r"(s2));
        asm("mad.lo.u32 %0, %0, %1, %2;" : "+r"(v) : "r"(s1), "r"(s2));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.2f cy/op=%.3f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS, (double)(t1-t0)/(double)ITERS/4.0);
    }
}
