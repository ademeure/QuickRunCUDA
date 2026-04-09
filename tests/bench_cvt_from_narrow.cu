// CVT Throughput: narrow format -> f16x2 (all formats)
// 16 independent CVTs per iteration, unrolled
// Define CVT_ASM via -H. For e2m1x2 (.b8 input), also define CVT_B8

#ifndef CVT_ASM
#define CVT_ASM cvt.rn.f16x2.e4m3x2
#endif
#ifndef UNROLL
#define UNROLL 8
#endif

#define _S(x) #x
#define S(x) _S(x)

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    unsigned short tid = (unsigned short)(threadIdx.x & 0xFF);
    // Thread-dependent base inputs
    const unsigned short base0  = 0x3838u ^ tid;
    const unsigned short base1  = 0x3839u ^ tid;
    const unsigned short base2  = 0x3938u ^ tid;
    const unsigned short base3  = 0x3939u ^ tid;
    const unsigned short base4  = 0x383Au ^ tid;
    const unsigned short base5  = 0x3A38u ^ tid;
    const unsigned short base6  = 0x3A39u ^ tid;
    const unsigned short base7  = 0x393Au ^ tid;
    const unsigned short base8  = 0x3A3Au ^ tid;
    const unsigned short base9  = 0x383Bu ^ tid;
    const unsigned short base10 = 0x3B38u ^ tid;
    const unsigned short base11 = 0x3B39u ^ tid;
    const unsigned short base12 = 0x393Bu ^ tid;
    const unsigned short base13 = 0x3B3Au ^ tid;
    const unsigned short base14 = 0x3A3Bu ^ tid;
    const unsigned short base15 = 0x3B3Bu ^ tid;

    unsigned int t0, t1, t2, t3, t4, t5, t6, t7;
    unsigned int t8, t9, t10, t11, t12, t13, t14, t15;
    unsigned int acc = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned short ci = (unsigned short)(i + j);
            unsigned short in0=base0^ci, in1=base1^ci, in2=base2^ci, in3=base3^ci;
            unsigned short in4=base4^ci, in5=base5^ci, in6=base6^ci, in7=base7^ci;
            unsigned short in8=base8^ci, in9=base9^ci, in10=base10^ci, in11=base11^ci;
            unsigned short in12=base12^ci, in13=base13^ci, in14=base14^ci, in15=base15^ci;

#ifdef CVT_B8
            asm volatile(
                "{ .reg .b8 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;\n\t"
                "mov.b16 {b0,_},%16; mov.b16 {b1,_},%17; mov.b16 {b2,_},%18; mov.b16 {b3,_},%19;\n\t"
                "mov.b16 {b4,_},%20; mov.b16 {b5,_},%21; mov.b16 {b6,_},%22; mov.b16 {b7,_},%23;\n\t"
                "mov.b16 {b8,_},%24; mov.b16 {b9,_},%25; mov.b16 {b10,_},%26; mov.b16 {b11,_},%27;\n\t"
                "mov.b16 {b12,_},%28; mov.b16 {b13,_},%29; mov.b16 {b14,_},%30; mov.b16 {b15,_},%31;\n\t"
                S(CVT_ASM) " %0,b0;\n\t"   S(CVT_ASM) " %1,b1;\n\t"
                S(CVT_ASM) " %2,b2;\n\t"   S(CVT_ASM) " %3,b3;\n\t"
                S(CVT_ASM) " %4,b4;\n\t"   S(CVT_ASM) " %5,b5;\n\t"
                S(CVT_ASM) " %6,b6;\n\t"   S(CVT_ASM) " %7,b7;\n\t"
                S(CVT_ASM) " %8,b8;\n\t"   S(CVT_ASM) " %9,b9;\n\t"
                S(CVT_ASM) " %10,b10;\n\t" S(CVT_ASM) " %11,b11;\n\t"
                S(CVT_ASM) " %12,b12;\n\t" S(CVT_ASM) " %13,b13;\n\t"
                S(CVT_ASM) " %14,b14;\n\t" S(CVT_ASM) " %15,b15;}\n"
                : "=r"(t0),"=r"(t1),"=r"(t2),"=r"(t3),"=r"(t4),"=r"(t5),"=r"(t6),"=r"(t7),
                  "=r"(t8),"=r"(t9),"=r"(t10),"=r"(t11),"=r"(t12),"=r"(t13),"=r"(t14),"=r"(t15)
                : "h"(in0),"h"(in1),"h"(in2),"h"(in3),"h"(in4),"h"(in5),"h"(in6),"h"(in7),
                  "h"(in8),"h"(in9),"h"(in10),"h"(in11),"h"(in12),"h"(in13),"h"(in14),"h"(in15)
            );
#else
            asm volatile(
                S(CVT_ASM) " %0,%16;\n\t"  S(CVT_ASM) " %1,%17;\n\t"
                S(CVT_ASM) " %2,%18;\n\t"  S(CVT_ASM) " %3,%19;\n\t"
                S(CVT_ASM) " %4,%20;\n\t"  S(CVT_ASM) " %5,%21;\n\t"
                S(CVT_ASM) " %6,%22;\n\t"  S(CVT_ASM) " %7,%23;\n\t"
                S(CVT_ASM) " %8,%24;\n\t"  S(CVT_ASM) " %9,%25;\n\t"
                S(CVT_ASM) " %10,%26;\n\t" S(CVT_ASM) " %11,%27;\n\t"
                S(CVT_ASM) " %12,%28;\n\t" S(CVT_ASM) " %13,%29;\n\t"
                S(CVT_ASM) " %14,%30;\n\t" S(CVT_ASM) " %15,%31;\n\t"
                : "=r"(t0),"=r"(t1),"=r"(t2),"=r"(t3),"=r"(t4),"=r"(t5),"=r"(t6),"=r"(t7),
                  "=r"(t8),"=r"(t9),"=r"(t10),"=r"(t11),"=r"(t12),"=r"(t13),"=r"(t14),"=r"(t15)
                : "h"(in0),"h"(in1),"h"(in2),"h"(in3),"h"(in4),"h"(in5),"h"(in6),"h"(in7),
                  "h"(in8),"h"(in9),"h"(in10),"h"(in11),"h"(in12),"h"(in13),"h"(in14),"h"(in15)
            );
#endif
            acc ^= t0^t1^t2^t3^t4^t5^t6^t7^t8^t9^t10^t11^t12^t13^t14^t15;
        }
    }

    if (threadIdx.x >= blockDim.x) { ((unsigned int*)C)[threadIdx.x] = acc; }
}
