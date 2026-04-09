// CVT Throughput: f16x2 -> narrow format (e4m3x2, e5m2x2, or e2m1x2)
// 16 independent CVTs per iteration, unrolled
// Define CVT_ASM via -H. For e2m1x2 (.b8 output), also define CVT_B8

#ifndef CVT_ASM
#define CVT_ASM cvt.rn.satfinite.e4m3x2.f16x2
#endif
#ifndef UNROLL
#define UNROLL 8
#endif

#define _S(x) #x
#define S(x) _S(x)

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    unsigned int tid = threadIdx.x;
    // Thread-dependent base inputs
    const unsigned int base0  = 0x3C003C00u ^ tid;
    const unsigned int base1  = 0x3C003C01u ^ tid;
    const unsigned int base2  = 0x3C013C00u ^ tid;
    const unsigned int base3  = 0x3C013C01u ^ tid;
    const unsigned int base4  = 0x3C023C00u ^ tid;
    const unsigned int base5  = 0x3C003C02u ^ tid;
    const unsigned int base6  = 0x3C023C01u ^ tid;
    const unsigned int base7  = 0x3C013C02u ^ tid;
    const unsigned int base8  = 0x3C023C02u ^ tid;
    const unsigned int base9  = 0x3C033C00u ^ tid;
    const unsigned int base10 = 0x3C003C03u ^ tid;
    const unsigned int base11 = 0x3C033C01u ^ tid;
    const unsigned int base12 = 0x3C013C03u ^ tid;
    const unsigned int base13 = 0x3C033C02u ^ tid;
    const unsigned int base14 = 0x3C023C03u ^ tid;
    const unsigned int base15 = 0x3C033C03u ^ tid;

    unsigned short t0, t1, t2, t3, t4, t5, t6, t7;
    unsigned short t8, t9, t10, t11, t12, t13, t14, t15;
    unsigned int acc = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            unsigned int ci = (unsigned int)(i + j);
            unsigned int in0=base0^ci, in1=base1^ci, in2=base2^ci, in3=base3^ci;
            unsigned int in4=base4^ci, in5=base5^ci, in6=base6^ci, in7=base7^ci;
            unsigned int in8=base8^ci, in9=base9^ci, in10=base10^ci, in11=base11^ci;
            unsigned int in12=base12^ci, in13=base13^ci, in14=base14^ci, in15=base15^ci;

#ifdef CVT_B8
            asm volatile(
                "{ .reg .b8 b0,b1,b2,b3,b4,b5,b6,b7,b8,b9,b10,b11,b12,b13,b14,b15;\n\t"
                S(CVT_ASM) " b0, %16;\n\t"  S(CVT_ASM) " b1, %17;\n\t"
                S(CVT_ASM) " b2, %18;\n\t"  S(CVT_ASM) " b3, %19;\n\t"
                S(CVT_ASM) " b4, %20;\n\t"  S(CVT_ASM) " b5, %21;\n\t"
                S(CVT_ASM) " b6, %22;\n\t"  S(CVT_ASM) " b7, %23;\n\t"
                S(CVT_ASM) " b8, %24;\n\t"  S(CVT_ASM) " b9, %25;\n\t"
                S(CVT_ASM) " b10, %26;\n\t" S(CVT_ASM) " b11, %27;\n\t"
                S(CVT_ASM) " b12, %28;\n\t" S(CVT_ASM) " b13, %29;\n\t"
                S(CVT_ASM) " b14, %30;\n\t" S(CVT_ASM) " b15, %31;\n\t"
                "mov.b16 %0,{b0,0}; mov.b16 %1,{b1,0}; mov.b16 %2,{b2,0}; mov.b16 %3,{b3,0};\n\t"
                "mov.b16 %4,{b4,0}; mov.b16 %5,{b5,0}; mov.b16 %6,{b6,0}; mov.b16 %7,{b7,0};\n\t"
                "mov.b16 %8,{b8,0}; mov.b16 %9,{b9,0}; mov.b16 %10,{b10,0}; mov.b16 %11,{b11,0};\n\t"
                "mov.b16 %12,{b12,0}; mov.b16 %13,{b13,0}; mov.b16 %14,{b14,0}; mov.b16 %15,{b15,0};}\n"
                : "=h"(t0),"=h"(t1),"=h"(t2),"=h"(t3),"=h"(t4),"=h"(t5),"=h"(t6),"=h"(t7),
                  "=h"(t8),"=h"(t9),"=h"(t10),"=h"(t11),"=h"(t12),"=h"(t13),"=h"(t14),"=h"(t15)
                : "r"(in0),"r"(in1),"r"(in2),"r"(in3),"r"(in4),"r"(in5),"r"(in6),"r"(in7),
                  "r"(in8),"r"(in9),"r"(in10),"r"(in11),"r"(in12),"r"(in13),"r"(in14),"r"(in15)
            );
#else
            asm volatile(
                S(CVT_ASM) " %0, %16;\n\t"  S(CVT_ASM) " %1, %17;\n\t"
                S(CVT_ASM) " %2, %18;\n\t"  S(CVT_ASM) " %3, %19;\n\t"
                S(CVT_ASM) " %4, %20;\n\t"  S(CVT_ASM) " %5, %21;\n\t"
                S(CVT_ASM) " %6, %22;\n\t"  S(CVT_ASM) " %7, %23;\n\t"
                S(CVT_ASM) " %8, %24;\n\t"  S(CVT_ASM) " %9, %25;\n\t"
                S(CVT_ASM) " %10, %26;\n\t" S(CVT_ASM) " %11, %27;\n\t"
                S(CVT_ASM) " %12, %28;\n\t" S(CVT_ASM) " %13, %29;\n\t"
                S(CVT_ASM) " %14, %30;\n\t" S(CVT_ASM) " %15, %31;\n\t"
                : "=h"(t0),"=h"(t1),"=h"(t2),"=h"(t3),"=h"(t4),"=h"(t5),"=h"(t6),"=h"(t7),
                  "=h"(t8),"=h"(t9),"=h"(t10),"=h"(t11),"=h"(t12),"=h"(t13),"=h"(t14),"=h"(t15)
                : "r"(in0),"r"(in1),"r"(in2),"r"(in3),"r"(in4),"r"(in5),"r"(in6),"r"(in7),
                  "r"(in8),"r"(in9),"r"(in10),"r"(in11),"r"(in12),"r"(in13),"r"(in14),"r"(in15)
            );
#endif
            acc ^= (unsigned int)(t0^t1^t2^t3^t4^t5^t6^t7^t8^t9^t10^t11^t12^t13^t14^t15);
        }
    }

    if (threadIdx.x >= blockDim.x) { ((unsigned int*)C)[threadIdx.x] = acc; }
}
