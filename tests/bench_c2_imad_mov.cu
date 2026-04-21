// C2: IMAD.MOV vs MOV throughput
// MODE 0: pure MOV via PTX mov.b32
// MODE 1: IMAD-as-MOV via mad.lo.u32 with *0 + src
// MODE 2: pure IADD3 add.u32 0 (via "add %0, %1, 0")
// MODE 3: pure IMAD with constant 1
// MODE 4: baseline LOP3 OR with 0
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(32, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    if (threadIdx.x >= 32) return;

    unsigned int v = (unsigned)(threadIdx.x ^ u2);
    unsigned int x = 0xDEADBEEFu ^ (unsigned)u2;

    unsigned long long t0, t1;
    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));

    #pragma unroll 1
    for (int i = 0; i < ITERS; i++) {
        #pragma unroll 16
        for (int u = 0; u < 16; u++) {
            // Make x change so MOV can't be eliminated
            x = v ^ x;
#if MODE == 0
            // mov.b32 (pure register copy)
            asm volatile("mov.b32 %0, %1;" : "=r"(v) : "r"(x));
#elif MODE == 1
            // IMAD.MOV: mad.lo.u32 %0, %1, 1, 0  (== src * 1 + 0 = src)
            asm volatile("mad.lo.u32 %0, %1, 1, 0;" : "=r"(v) : "r"(x));
#elif MODE == 2
            // add.u32 with 0 (IADD3-as-MOV)
            asm volatile("add.u32 %0, %1, 0;" : "=r"(v) : "r"(x));
#elif MODE == 3
            // mad.lo.u32 with multiplier 1, addend in register (IMAD chain — Cluster A)
            asm volatile("mad.lo.u32 %0, %1, 1, %1;" : "=r"(v) : "r"(x));
#elif MODE == 4
            // lop3.b32 with src OR 0 (= src) (Cluster B)
            asm volatile("lop3.b32 %0, %1, 0, 0, 0xFC;" : "=r"(v) : "r"(x));
#endif
        }
    }

    asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));

    if (v == (unsigned)seed) ((unsigned*)C)[blockIdx.x] = v;
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        printf("MODE=%d clk=%llu cy/iter=%.3f cy/op=%.4f\n",
               MODE, t1-t0, (double)(t1-t0)/(double)ITERS,
               (double)(t1-t0)/(double)ITERS/16.0);
    }
}
