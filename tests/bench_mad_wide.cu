// mad vs mad.wide vs mad.hi SASS comparison

#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    int* p = (int*)A;
    long long* lp = (long long*)A;
    int idx = (threadIdx.x + blockIdx.x * blockDim.x) & 0x3FF;
    unsigned int a = (unsigned)idx;
    unsigned int b = (unsigned)idx + 1;
    unsigned int c = (unsigned)idx + 2;
    unsigned long long lc = (unsigned long long)c << 32;

#if MODE == 0
    // mad.lo.u32 (return low 32 bits)
    unsigned int r;
    asm("mad.lo.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    p[idx] = r;
#elif MODE == 1
    // mad.hi.u32 (return high 32 bits of multiply)
    unsigned int r;
    asm("mad.hi.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    p[idx] = r;
#elif MODE == 2
    // mad.wide.u32 (32x32 -> 64, plus 64-bit add)
    unsigned long long r;
    asm("mad.wide.u32 %0, %1, %2, %3;" : "=l"(r) : "r"(a), "r"(b), "l"(lc));
    lp[idx] = r;
#elif MODE == 3
    // mad.lo.cc.u32 (carry-out)
    unsigned int r;
    asm("{ .reg .pred p; mad.lo.cc.u32 %0, %1, %2, %3; }" : "=r"(r) : "r"(a), "r"(b), "r"(c));
    p[idx] = r;
#endif
}
