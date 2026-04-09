// CVT Throughput: f32 x4 -> narrow format with stochastic rounding (cvt.rs)
// 4 independent CVTs per iteration, unrolled
// Define CVT_ASM via -H. For e2m1x4 (.b16 output): default. Others: define CVT_B32

#ifndef CVT_ASM
#define CVT_ASM cvt.rs.satfinite.e4m3x4.f32
#endif
#ifndef UNROLL
#define UNROLL 8
#endif

#define _S(x) #x
#define S(x) _S(x)

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    float tid_f = (float)(threadIdx.x & 0xFF) * 0.001f;
    unsigned int rbits = threadIdx.x * 2654435761u + 1;
    unsigned int acc = 0;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            float fi = __int_as_float((i + j) ^ threadIdx.x);
            float a0=1.0f+fi+tid_f, a1=2.0f+fi, a2=0.5f+fi+tid_f, a3=1.5f+fi;
            float b0=3.0f+fi, b1=0.25f+fi+tid_f, b2=4.0f+fi, b3=0.125f+fi+tid_f;
            float c0=1.0f-fi+tid_f, c1=2.0f-fi, c2=0.5f-fi+tid_f, c3=1.5f-fi;
            float d0=3.0f-fi, d1=0.25f-fi+tid_f, d2=4.0f-fi, d3=0.125f-fi+tid_f;
            rbits = rbits * 1664525u + 1013904223u;

#ifdef CVT_B32
            unsigned int t0, t1, t2, t3;
            asm volatile(
                S(CVT_ASM) " %0, {%4,%5,%6,%7}, %16;\n\t"
                S(CVT_ASM) " %1, {%8,%9,%10,%11}, %16;\n\t"
                S(CVT_ASM) " %2, {%12,%13,%14,%15}, %16;\n\t"
                S(CVT_ASM) " %3, {%17,%18,%19,%20}, %16;\n\t"
                : "=r"(t0),"=r"(t1),"=r"(t2),"=r"(t3)
                : "f"(a0),"f"(a1),"f"(a2),"f"(a3),
                  "f"(b0),"f"(b1),"f"(b2),"f"(b3),
                  "f"(c0),"f"(c1),"f"(c2),"f"(c3),
                  "r"(rbits),
                  "f"(d0),"f"(d1),"f"(d2),"f"(d3)
            );
            acc ^= t0^t1^t2^t3;
#else
            unsigned short t0, t1, t2, t3;
            asm volatile(
                "{ .reg .b16 tmp0,tmp1,tmp2,tmp3;\n\t"
                S(CVT_ASM) " tmp0, {%4,%5,%6,%7}, %16;\n\t"
                S(CVT_ASM) " tmp1, {%8,%9,%10,%11}, %16;\n\t"
                S(CVT_ASM) " tmp2, {%12,%13,%14,%15}, %16;\n\t"
                S(CVT_ASM) " tmp3, {%17,%18,%19,%20}, %16;\n\t"
                "mov.b16 %0,tmp0; mov.b16 %1,tmp1; mov.b16 %2,tmp2; mov.b16 %3,tmp3;}\n"
                : "=h"(t0),"=h"(t1),"=h"(t2),"=h"(t3)
                : "f"(a0),"f"(a1),"f"(a2),"f"(a3),
                  "f"(b0),"f"(b1),"f"(b2),"f"(b3),
                  "f"(c0),"f"(c1),"f"(c2),"f"(c3),
                  "r"(rbits),
                  "f"(d0),"f"(d1),"f"(d2),"f"(d3)
            );
            acc ^= (unsigned int)(t0^t1^t2^t3);
#endif
        }
    }

    if (threadIdx.x >= blockDim.x) { ((unsigned int*)C)[threadIdx.x] = acc; }
}
