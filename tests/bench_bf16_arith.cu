// BF16 scalar / vec operations — full catalog.
// Discovered missing: scalar BF16 add/mul/fma, BF16 compare, BF16 cvt chains.

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
    unsigned int x[8];
    unsigned short xs[8];
    #pragma unroll
    for (int k=0;k<8;k++) { x[k] = 0x3F803F80u ^ (k<<4); xs[k] = 0x3F80 ^ k; }

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k=0;k<8;k++) {
                unsigned int nxt = x[(k+1) & 7];
                unsigned short nxts = xs[(k+1) & 7];
#if OP == 0   // bf16x2 fma
                asm volatile("fma.rn.bf16x2 %0, %0, %1, %1;" : "+r"(x[k]) : "r"(nxt));
#elif OP == 1 // bf16x2 add
                asm volatile("add.rn.bf16x2 %0, %0, %1;" : "+r"(x[k]) : "r"(nxt));
#elif OP == 2 // bf16x2 mul
                asm volatile("mul.rn.bf16x2 %0, %0, %1;" : "+r"(x[k]) : "r"(nxt));
#elif OP == 3 // bf16x2 min
                asm volatile("min.bf16x2 %0, %0, %1;" : "+r"(x[k]) : "r"(nxt));
#elif OP == 4 // bf16x2 abs
                asm volatile("abs.bf16x2 %0, %0;" : "+r"(x[k]));
#elif OP == 5 // bf16x2 neg
                asm volatile("neg.bf16x2 %0, %0;" : "+r"(x[k]));
#elif OP == 6 // scalar bf16 add (if exists)
                asm volatile("add.rn.bf16 %0, %0, %1;" : "+h"(xs[k]) : "h"(nxts));
#elif OP == 7 // scalar bf16 fma
                asm volatile("fma.rn.bf16 %0, %0, %1, %1;" : "+h"(xs[k]) : "h"(nxts));
#elif OP == 8 // bf16x2 setp + selp
                unsigned int r;
                asm volatile("{.reg .pred p0,p1; setp.lt.bf16x2 p0|p1, %1, %2; selp.b32 %0, %1, 0, p0;}"
                    : "=r"(r) : "r"(x[k]), "r"(nxt));
                x[k] = r;
#elif OP == 9 // cvt bf16 -> f32 pair (broadcast packed)
                float fa, fb;
                asm volatile("{.reg .bf16 a,b; mov.b32 {a,b}, %2; cvt.f32.bf16 %0, a; cvt.f32.bf16 %1, b;}"
                    : "=f"(fa), "=f"(fb) : "r"(x[k]));
                x[k] = __float_as_int(fa + fb);
#elif OP == 10 // cvt f32 pair -> bf16x2 (pack)
                asm volatile("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(x[k])
                    : "f"(__int_as_float(x[k])), "f"(__int_as_float(nxt)));
#endif
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k=0;k<8;k++) acc ^= x[k] ^ (unsigned int)xs[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
