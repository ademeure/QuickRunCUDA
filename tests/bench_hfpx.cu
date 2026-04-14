// Half-precision specialized ops: HSET2, HSETP2, HADD2 with different rounding.

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
    #pragma unroll
    for (int k=0;k<8;k++) v[k] = 0x3C003C01u ^ (threadIdx.x*131 + k*17);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<8;k++) {
                unsigned int nxt = v[(k+1)&7];
#if OP == 0  // add.rn.f16 (scalar half)
                unsigned short a = (unsigned short)v[k], b = (unsigned short)nxt, c;
                asm volatile("add.rn.f16 %0, %1, %2;" : "=h"(c) : "h"(a), "h"(b));
                v[k] = (unsigned int)c;
#elif OP == 1  // fma.rn.f16 scalar
                unsigned short a = (unsigned short)v[k], b = (unsigned short)nxt, c;
                asm volatile("fma.rn.f16 %0, %1, %2, %2;" : "=h"(c) : "h"(a), "h"(b));
                v[k] = (unsigned int)c;
#elif OP == 2  // add.rn.bf16 scalar
                unsigned short a = (unsigned short)v[k], b = (unsigned short)nxt, c;
                asm volatile("add.rn.bf16 %0, %1, %2;" : "=h"(c) : "h"(a), "h"(b));
                v[k] = (unsigned int)c;
#elif OP == 3  // fma.rn.bf16 scalar
                unsigned short a = (unsigned short)v[k], b = (unsigned short)nxt, c;
                asm volatile("fma.rn.bf16 %0, %1, %2, %2;" : "=h"(c) : "h"(a), "h"(b));
                v[k] = (unsigned int)c;
#elif OP == 4  // setp.eq.f16 scalar
                unsigned short a = (unsigned short)v[k], b = (unsigned short)nxt;
                unsigned int r;
                asm volatile("{.reg .pred p; setp.eq.f16 p, %1, %2; selp.u32 %0, 1, 2, p;}" : "=r"(r) : "h"(a), "h"(b));
                v[k] = r;
#elif OP == 5  // neg.f16x2
                asm volatile("neg.f16x2 %0, %0;" : "+r"(v[k]));
#elif OP == 6  // neg.bf16x2
                asm volatile("neg.bf16x2 %0, %0;" : "+r"(v[k]));
#elif OP == 7  // abs.f16x2
                asm volatile("abs.f16x2 %0, %0;" : "+r"(v[k]));
#elif OP == 8  // tanh.approx.f16 (Hopper+)
                unsigned short a = (unsigned short)v[k], c;
                asm volatile("tanh.approx.f16 %0, %1;" : "=h"(c) : "h"(a));
                v[k] = (unsigned int)c;
#elif OP == 9  // tanh.approx.f16x2
                asm volatile("tanh.approx.f16x2 %0, %0;" : "+r"(v[k]));
#elif OP == 10  // ex2.approx.f16
                unsigned short a = (unsigned short)v[k], c;
                asm volatile("ex2.approx.f16 %0, %1;" : "=h"(c) : "h"(a));
                v[k] = (unsigned int)c;
#elif OP == 11  // ex2.approx.f16x2
                asm volatile("ex2.approx.f16x2 %0, %0;" : "+r"(v[k]));
#endif
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k=0;k<8;k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
