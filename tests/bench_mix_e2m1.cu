// Instruction mix benchmark: e2m1x2 CVTs mixed with companion instructions
// Define N_CVT, N_COMP, COMP_ASM, COMP_CONSTRAINT via -H
// Measures impact of mixing F2FP with FFMA/FMUL/LOP3/IADD/IMAD/MUFU on throughput

#ifndef N_CVT
#define N_CVT 4
#endif
#ifndef N_COMP
#define N_COMP 0
#endif
#ifndef UNROLL
#define UNROLL 4
#endif

// Companion instruction defaults (FFMA)
#ifndef COMP_INIT
#define COMP_INIT float c0=1.0f+tid_f, c1=2.0f+tid_f, c2=3.0f+tid_f, c3=4.0f+tid_f; \
    float ca=1.0000001f, cb=0.9999999f;
#endif
#ifndef COMP_1
#define COMP_1 asm volatile("fma.rn.f32 %0,%1,%2,%0;" : "+f"(c0) : "f"(ca),"f"(cb));
#endif
#ifndef COMP_SINK
#define COMP_SINK acc ^= __float_as_int(c0) ^ __float_as_int(c1) ^ __float_as_int(c2) ^ __float_as_int(c3);
#endif

// CVT on 4 different inputs (rotated)
#define CVT_1(idx) asm volatile( \
    "{ .reg .b8 _t; cvt.rn.satfinite.e2m1x2.f16x2 _t, %1; mov.b16 %0,{_t,0}; }" \
    : "=h"(h0) : "r"(ins[idx & 3]));

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    unsigned int tid = threadIdx.x;
    float tid_f = (float)(tid & 0xFF) * 0.001f;

    unsigned int ins[4] = {
        0x3C003C00u ^ tid, 0x3C013C01u ^ tid,
        0x3C023C02u ^ tid, 0x3C033C03u ^ tid
    };

    unsigned short h0 = 0;
    unsigned int acc = 0;

    COMP_INIT

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Vary CVT inputs per iteration
            ins[0] ^= (unsigned int)(i + j);
            ins[1] ^= (unsigned int)(i + j + 1);
            ins[2] ^= (unsigned int)(i + j + 2);
            ins[3] ^= (unsigned int)(i + j + 3);

            // Interleaved CVT and companion instructions
            // Compiler will emit them in order; ptxas schedules freely
#if N_CVT >= 1
            CVT_1(0)  acc ^= h0;
#endif
#if N_COMP >= 1
            COMP_1
#endif
#if N_CVT >= 2
            CVT_1(1)  acc ^= h0;
#endif
#if N_COMP >= 2
            COMP_1
#endif
#if N_CVT >= 3
            CVT_1(2)  acc ^= h0;
#endif
#if N_COMP >= 3
            COMP_1
#endif
#if N_CVT >= 4
            CVT_1(3)  acc ^= h0;
#endif
#if N_COMP >= 4
            COMP_1
#endif
            // Extra companions beyond 4 (added in blocks of 4)
#if N_COMP >= 5
            COMP_1
#endif
#if N_COMP >= 6
            COMP_1
#endif
#if N_COMP >= 7
            COMP_1
#endif
#if N_COMP >= 8
            COMP_1
#endif
#if N_COMP >= 9
            COMP_1
#endif
#if N_COMP >= 10
            COMP_1
#endif
#if N_COMP >= 11
            COMP_1
#endif
#if N_COMP >= 12
            COMP_1
#endif
#if N_COMP >= 13
            COMP_1
#endif
#if N_COMP >= 14
            COMP_1
#endif
#if N_COMP >= 15
            COMP_1
#endif
#if N_COMP >= 16
            COMP_1
#endif
            // Extra CVTs beyond 4
#if N_CVT >= 5
            CVT_1(0)  acc ^= h0;
#endif
#if N_CVT >= 6
            CVT_1(1)  acc ^= h0;
#endif
#if N_CVT >= 7
            CVT_1(2)  acc ^= h0;
#endif
#if N_CVT >= 8
            CVT_1(3)  acc ^= h0;
#endif
#if N_CVT >= 9
            CVT_1(0)  acc ^= h0;
#endif
#if N_CVT >= 10
            CVT_1(1)  acc ^= h0;
#endif
#if N_CVT >= 11
            CVT_1(2)  acc ^= h0;
#endif
#if N_CVT >= 12
            CVT_1(3)  acc ^= h0;
#endif
#if N_CVT >= 13
            CVT_1(0)  acc ^= h0;
#endif
#if N_CVT >= 14
            CVT_1(1)  acc ^= h0;
#endif
#if N_CVT >= 15
            CVT_1(2)  acc ^= h0;
#endif
#if N_CVT >= 16
            CVT_1(3)  acc ^= h0;
#endif
        }
    }

    COMP_SINK

    if (tid >= (unsigned int)blockDim.x) {
        ((unsigned int*)C)[tid] = acc;
    }
}
