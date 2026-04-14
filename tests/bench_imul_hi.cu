// IMUL.HI vs IMAD.HI, and more integer multiply variants.

#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 4
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[8];
    int s[8];
    #pragma unroll
    for (int k=0;k<8;k++) { v[k] = threadIdx.x + k; s[k] = (int)v[k]; }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<8;k++) {
                unsigned int n = v[(k+1)&7];
#if OP == 0  // mul.lo.u32
                asm volatile("mul.lo.u32 %0, %0, %1;" : "+r"(v[k]) : "r"(n));
#elif OP == 1  // mul.hi.u32
                asm volatile("mul.hi.u32 %0, %0, %1;" : "+r"(v[k]) : "r"(n));
#elif OP == 2  // mul.hi.s32
                asm volatile("mul.hi.s32 %0, %0, %1;" : "+r"(s[k]) : "r"((int)n));
                v[k] = (unsigned)s[k];
#elif OP == 3  // mad.hi.u32
                asm volatile("mad.hi.u32 %0, %0, %1, %2;" : "+r"(v[k]) : "r"(n), "r"(n));
#elif OP == 4  // mul.wide.u16
                unsigned short a = (unsigned short)v[k], b = (unsigned short)n;
                asm volatile("mul.wide.u16 %0, %1, %2;" : "=r"(v[k]) : "h"(a), "h"(b));
#elif OP == 5  // mad.wide.u16
                unsigned short a = (unsigned short)v[k], b = (unsigned short)n;
                asm volatile("mad.wide.u16 %0, %1, %2, %0;" : "+r"(v[k]) : "h"(a), "h"(b));
#elif OP == 6  // mul.wide.u32 (u32 × u32 → u64)
                unsigned long long w;
                asm volatile("mul.wide.u32 %0, %1, %2;" : "=l"(w) : "r"(v[k]), "r"(n));
                v[k] = (unsigned)w;
#elif OP == 7  // mad.lo.u16
                unsigned short a = (unsigned short)v[k], b = (unsigned short)n, c = (unsigned short)v[(k+2)&7], r;
                asm volatile("mad.lo.u16 %0, %1, %2, %3;" : "=h"(r) : "h"(a), "h"(b), "h"(c));
                v[k] = (unsigned int)r;
#elif OP == 8  // add.u16 (scalar half-word add)
                unsigned short a = (unsigned short)v[k], b = (unsigned short)n, r;
                asm volatile("add.u16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b));
                v[k] = (unsigned int)r;
#elif OP == 9  // max.u16 / min.u16
                unsigned short a = (unsigned short)v[k], b = (unsigned short)n, r;
                asm volatile("max.u16 %0, %1, %2;" : "=h"(r) : "h"(a), "h"(b));
                v[k] = (unsigned int)r;
#endif
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k=0;k<8;k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
