// F2FP SCALAR (MERGE_C) + companion co-issue. Test specifically whether
// scalar F2FP (cvt.rn.satfinite.f16.f32 — compiles to F2FP.SATFINITE.F16.F32.MERGE_C)
// contends with LOP3. MERGE_C does an internal read-merge on the destination,
// which might share the LOP3 pipe.

#ifndef N_CVT
#define N_CVT 16
#endif
#ifndef N_COMP
#define N_COMP 0
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif
#ifndef COMP_TYPE
#define COMP_TYPE 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    float r[N_CVT];
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) r[k] = (float)(threadIdx.x + k + 1) * 1.00001f;

#if COMP_TYPE == 0 || COMP_TYPE == 1 || COMP_TYPE == 4
    float c[N_COMP > 0 ? N_COMP : 1];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) c[m] = (float)(threadIdx.x + m + 1) * 1.0001f;
    const float ca = 1.0000001f;
#else
    unsigned int c[N_COMP > 0 ? N_COMP : 1];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) c[m] = threadIdx.x + m + 0xBEEF;
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Scalar F2FP: cvt.rn.satfinite.f16.f32 (compiler will fold the widen,
            // leaving just one F2FP.MERGE_C per k per j)
            #pragma unroll
            for (int k = 0; k < N_CVT; k++) {
                unsigned short h;
                asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(h) : "f"(r[k]));
                asm volatile("xor.b16 %0, %0, 1;" : "+h"(h));
                asm volatile("cvt.f32.f16 %0, %1;" : "=f"(r[k]) : "h"(h));
                // The xor forces the chain to not fully collapse; the cvt.f32.f16
                // widen gets folded into register. Per-k net: 1 F2FP + 1 LOP3 (xor).
            }
            #pragma unroll
            for (int m = 0; m < N_COMP; m++) {
#if COMP_TYPE == 0
                asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(c[m]) : "f"(ca));
#elif COMP_TYPE == 1
                asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(c[m]) : "f"(ca));
#elif COMP_TYPE == 2
                asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(c[m]));
#elif COMP_TYPE == 3
                asm volatile("xor.b32 %0, %0, 0x1;" : "+r"(c[m]));
#elif COMP_TYPE == 4
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(c[m]));
#elif COMP_TYPE == 5
                asm volatile("shl.b32 %0, %0, 1;" : "+r"(c[m]));
#elif COMP_TYPE == 6
                asm volatile("add.u32 %0, %0, 1;" : "+r"(c[m]));
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) acc ^= __float_as_int(r[k]);
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) {
#if COMP_TYPE == 0 || COMP_TYPE == 1 || COMP_TYPE == 4
        acc ^= __float_as_int(c[m]);
#else
        acc ^= c[m];
#endif
    }
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
