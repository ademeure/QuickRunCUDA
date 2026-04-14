// Vector (per-lane) integer op latencies — force divergent values.

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
    if (blockIdx.x != 0) return;  // but keep all lanes of warp 0 active
    // Per-lane distinct starting value → vector pipe
    unsigned int x = 0x12345678u ^ threadIdx.x;
    unsigned int y = 0xABCDEF01u ^ (threadIdx.x * 37);
    unsigned long long l = ((unsigned long long)threadIdx.x << 16) | 0xDEADBEEFULL;
    unsigned long long total_dt = 0;

    #pragma unroll 1
    for (int outer = 0; outer < ITERS_OUTER; outer++) {
        unsigned long long t0, t1;
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t0));
        #pragma unroll
        for (int j = 0; j < N_OPS; j++) {
#if OP == 0
            asm volatile("mad.lo.u32 %0, %0, %1, 1;" : "+r"(x) : "r"(y));
#elif OP == 1
            asm volatile("add.u32 %0, %0, %1;" : "+r"(x) : "r"(y));
#elif OP == 2
            asm volatile("mul.hi.u32 %0, %0, %1;" : "+r"(x) : "r"(y));
#elif OP == 3
            asm volatile("xor.b32 %0, %0, %1;" : "+r"(x) : "r"(y));
#elif OP == 4
            asm volatile("prmt.b32 %0, %0, %0, %1;" : "+r"(x) : "r"(y & 0xFFFFu));
#elif OP == 5
            asm volatile("shf.l.wrap.b32 %0, %0, %0, %1;" : "+r"(x) : "r"(y & 0x1Fu));
#elif OP == 6
            asm volatile("brev.b32 %0, %0;" : "+r"(x));
#elif OP == 7
            unsigned int r;
            asm volatile("popc.b32 %0, %1;" : "=r"(r) : "r"(x));
            x = r ^ y;
#elif OP == 8
            unsigned int r;
            asm volatile("bfind.u32 %0, %1;" : "=r"(r) : "r"(x | 1));
            x = r + y;
#elif OP == 9
            asm volatile("add.u64 %0, %0, %1;" : "+l"(l) : "l"((unsigned long long)y));
            x = (unsigned int)l;
#elif OP == 10
            asm volatile("min.u32 %0, %0, %1;" : "+r"(x) : "r"(y));
#endif
            y = y * 3 + 1;  // keep y data-dependent
        }
        asm volatile("mov.u64 %0, %%clock64;" : "=l"(t1));
        total_dt += (t1 - t0);
    }
    if ((int)x == seed) ((unsigned int*)C)[0] = x + (unsigned)l;
    ((unsigned long long*)C)[1] = total_dt;
}
