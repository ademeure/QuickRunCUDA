// A1: Per-SMSP dual-issue: can FFMA (Cluster A) issue same cycle as LOP3 (Cluster B)?
// 4 independent chains per type; if dual-issue, mix runs in max(FFMA, LOP3) time.
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    // 4 indep FFMA chains
    float a0=(float)threadIdx.x*1.001f+(float)u2*1e-9f, a1=a0+0.1f, a2=a0+0.2f, a3=a0+0.3f;
    float ya=(float)(threadIdx.x^u2)*0.001f+1.0f, za=0.5f;

    // 4 indep LOP3 chains
    unsigned int b0=(unsigned)threadIdx.x^(unsigned)u2, b1=b0+1, b2=b0+2, b3=b0+3;
    unsigned int yb=0xDEADBEEFu, zb=0xCAFEBABEu;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // 4 indep FFMA only
        a0 = a0*ya + za; a1 = a1*ya + za; a2 = a2*ya + za; a3 = a3*ya + za;
#elif MODE == 1
        // 4 indep LOP3 only
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b2) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b3) : "r"(yb), "r"(zb));
#elif MODE == 2
        // Mixed: 4 FFMA + 4 LOP3 (interleaved in code)
        a0 = a0*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(zb));
        a1 = a1*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(zb));
        a2 = a2*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b2) : "r"(yb), "r"(zb));
        a3 = a3*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b3) : "r"(yb), "r"(zb));
#elif MODE == 3
        // 8 indep FFMA only — measure throughput limit Cluster A
        a0 = a0*ya + za; a1 = a1*ya + za; a2 = a2*ya + za; a3 = a3*ya + za;
        a0 = a0*ya + za; a1 = a1*ya + za; a2 = a2*ya + za; a3 = a3*ya + za;
#elif MODE == 4
        // 8 indep LOP3 only
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b2) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b3) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b2) : "r"(yb), "r"(zb));
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b3) : "r"(yb), "r"(zb));
#elif MODE == 5
        // 8 FFMA + 8 LOP3 mixed
        a0 = a0*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(zb));
        a1 = a1*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(zb));
        a2 = a2*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b2) : "r"(yb), "r"(zb));
        a3 = a3*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b3) : "r"(yb), "r"(zb));
        a0 = a0*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b0) : "r"(yb), "r"(zb));
        a1 = a1*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b1) : "r"(yb), "r"(zb));
        a2 = a2*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b2) : "r"(yb), "r"(zb));
        a3 = a3*ya + za;
        asm("lop3.b32 %0, %0, %1, %2, 0xC8;" : "+r"(b3) : "r"(yb), "r"(zb));
#elif MODE == 6
        // FFMA + IMAD (both Cluster A) — should NOT dual-issue
        a0 = a0*ya + za; a1 = a1*ya + za; a2 = a2*ya + za; a3 = a3*ya + za;
        asm("mad.lo.u32 %0, %0, %1, %2;" : "+r"(b0) : "r"(yb), "r"(zb));
        asm("mad.lo.u32 %0, %0, %1, %2;" : "+r"(b1) : "r"(yb), "r"(zb));
        asm("mad.lo.u32 %0, %0, %1, %2;" : "+r"(b2) : "r"(yb), "r"(zb));
        asm("mad.lo.u32 %0, %0, %1, %2;" : "+r"(b3) : "r"(yb), "r"(zb));
#endif
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    float sink = a0+a1+a2+a3 + (float)(b0^b1^b2^b3);
    if ((int)sink == seed) C[blockIdx.x] = sink;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f\n", MODE, t1-t0, (double)(t1-t0)/(double)ITERS);
    }
}
