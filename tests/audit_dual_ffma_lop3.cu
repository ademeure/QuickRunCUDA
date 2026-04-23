// Audit: dual-issue FFMA + LOP3 interleaved.
// Per inner slot: one FFMA + one LOP3 (8 of each per UNROLL block).
// Anti-DCE: store accumulator under impossible predicate.

#ifndef UNROLL
#define UNROLL 8
#endif

extern "C" __global__ void kernel(float* A, float* B, float* C, int ITERS, int unused_1, int unused_2) {
    float tid_f = __int_as_float(threadIdx.x | 0x3F800000u); // ~1.x
    float a = 1.0000001f;
    float b = 0.9999999f;
    float r0 = tid_f, r1 = tid_f + 1.0f, r2 = tid_f + 2.0f, r3 = tid_f + 3.0f;
    float r4 = tid_f + 4.0f, r5 = tid_f + 5.0f, r6 = tid_f + 6.0f, r7 = tid_f + 7.0f;

    unsigned u0 = threadIdx.x * 7u + 0;
    unsigned u1 = threadIdx.x * 7u + 13;
    unsigned u2 = threadIdx.x * 7u + 26;
    unsigned u3 = threadIdx.x * 7u + 39;
    unsigned u4 = threadIdx.x * 7u + 52;
    unsigned u5 = threadIdx.x * 7u + 65;
    unsigned u6 = threadIdx.x * 7u + 78;
    unsigned u7 = threadIdx.x * 7u + 91;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            asm volatile(
                "fma.rn.f32 %0, %16, %17, %0;\n\t"
                "lop3.b32 %8, %8, 0xa5a5a5a5, 0x12345678, 0x96;\n\t"
                "fma.rn.f32 %1, %16, %17, %1;\n\t"
                "lop3.b32 %9, %9, 0xa5a5a5a5, 0x12345678, 0x96;\n\t"
                "fma.rn.f32 %2, %16, %17, %2;\n\t"
                "lop3.b32 %10, %10, 0xa5a5a5a5, 0x12345678, 0x96;\n\t"
                "fma.rn.f32 %3, %16, %17, %3;\n\t"
                "lop3.b32 %11, %11, 0xa5a5a5a5, 0x12345678, 0x96;\n\t"
                "fma.rn.f32 %4, %16, %17, %4;\n\t"
                "lop3.b32 %12, %12, 0xa5a5a5a5, 0x12345678, 0x96;\n\t"
                "fma.rn.f32 %5, %16, %17, %5;\n\t"
                "lop3.b32 %13, %13, 0xa5a5a5a5, 0x12345678, 0x96;\n\t"
                "fma.rn.f32 %6, %16, %17, %6;\n\t"
                "lop3.b32 %14, %14, 0xa5a5a5a5, 0x12345678, 0x96;\n\t"
                "fma.rn.f32 %7, %16, %17, %7;\n\t"
                "lop3.b32 %15, %15, 0xa5a5a5a5, 0x12345678, 0x96;\n\t"
                : "+f"(r0), "+f"(r1), "+f"(r2), "+f"(r3),
                  "+f"(r4), "+f"(r5), "+f"(r6), "+f"(r7),
                  "+r"(u0), "+r"(u1), "+r"(u2), "+r"(u3),
                  "+r"(u4), "+r"(u5), "+r"(u6), "+r"(u7)
                : "f"(a), "f"(b)
            );
        }
    }

    if (threadIdx.x >= blockDim.x) {
        C[threadIdx.x] = r0+r1+r2+r3+r4+r5+r6+r7
                       + __uint_as_float(u0^u1^u2^u3^u4^u5^u6^u7);
    }
}
