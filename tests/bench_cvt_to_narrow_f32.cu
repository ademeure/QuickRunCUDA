// CVT Throughput: f32 pair -> narrow format (all formats)
// 8 independent CVTs per iteration, unrolled
// Define CVT_ASM via -H. For e2m1x2 (.b8 output), also define CVT_B8

#ifndef CVT_ASM
#define CVT_ASM cvt.rn.satfinite.e4m3x2.f32
#endif
#ifndef UNROLL
#define UNROLL 8
#endif

#define _S(x) #x
#define S(x) _S(x)

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    // Thread-dependent base values
    float tid_f = (float)(threadIdx.x & 0xFF) * 0.001f;
    unsigned int acc = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            float fi = __int_as_float((i + j) ^ threadIdx.x);
            float a0 = 1.0f + fi + tid_f, b0 = 2.0f + fi;
            float a1 = 0.5f + fi, b1 = 1.5f + fi + tid_f;
            float a2 = 3.0f + fi + tid_f, b2 = 0.25f + fi;
            float a3 = 4.0f + fi, b3 = 0.125f + fi + tid_f;
            float a4 = 1.5f + fi + tid_f, b4 = 2.5f + fi;
            float a5 = 0.75f + fi, b5 = 1.25f + fi + tid_f;
            float a6 = 3.5f + fi + tid_f, b6 = 0.375f + fi;
            float a7 = 4.5f + fi, b7 = 0.0625f + fi + tid_f;

            unsigned short t0, t1, t2, t3, t4, t5, t6, t7;
#ifdef CVT_B8
            asm volatile(
                "{ .reg .b8 c0,c1,c2,c3,c4,c5,c6,c7;\n\t"
                S(CVT_ASM) " c0, %9, %8;\n\t"  S(CVT_ASM) " c1, %11, %10;\n\t"
                S(CVT_ASM) " c2, %13, %12;\n\t" S(CVT_ASM) " c3, %15, %14;\n\t"
                S(CVT_ASM) " c4, %17, %16;\n\t" S(CVT_ASM) " c5, %19, %18;\n\t"
                S(CVT_ASM) " c6, %21, %20;\n\t" S(CVT_ASM) " c7, %23, %22;\n\t"
                "mov.b16 %0,{c0,0}; mov.b16 %1,{c1,0}; mov.b16 %2,{c2,0}; mov.b16 %3,{c3,0};\n\t"
                "mov.b16 %4,{c4,0}; mov.b16 %5,{c5,0}; mov.b16 %6,{c6,0}; mov.b16 %7,{c7,0};}\n"
                : "=h"(t0),"=h"(t1),"=h"(t2),"=h"(t3),"=h"(t4),"=h"(t5),"=h"(t6),"=h"(t7)
                : "f"(a0),"f"(b0),"f"(a1),"f"(b1),"f"(a2),"f"(b2),"f"(a3),"f"(b3),
                  "f"(a4),"f"(b4),"f"(a5),"f"(b5),"f"(a6),"f"(b6),"f"(a7),"f"(b7)
            );
#else
            asm volatile(
                S(CVT_ASM) " %0, %9, %8;\n\t"  S(CVT_ASM) " %1, %11, %10;\n\t"
                S(CVT_ASM) " %2, %13, %12;\n\t" S(CVT_ASM) " %3, %15, %14;\n\t"
                S(CVT_ASM) " %4, %17, %16;\n\t" S(CVT_ASM) " %5, %19, %18;\n\t"
                S(CVT_ASM) " %6, %21, %20;\n\t" S(CVT_ASM) " %7, %23, %22;\n\t"
                : "=h"(t0),"=h"(t1),"=h"(t2),"=h"(t3),"=h"(t4),"=h"(t5),"=h"(t6),"=h"(t7)
                : "f"(a0),"f"(b0),"f"(a1),"f"(b1),"f"(a2),"f"(b2),"f"(a3),"f"(b3),
                  "f"(a4),"f"(b4),"f"(a5),"f"(b5),"f"(a6),"f"(b6),"f"(a7),"f"(b7)
            );
#endif
            acc ^= (unsigned int)(t0^t1^t2^t3^t4^t5^t6^t7);
        }
    }

    if (threadIdx.x >= blockDim.x) { ((unsigned int*)C)[threadIdx.x] = acc; }
}
