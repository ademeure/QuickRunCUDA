// F2FP + companion co-issue contention test.
//
// Run F2FP (packed narrow round-trip) at or near peak throughput, then add M
// companion ops per inner iteration. If companion pipe shares issue slot with
// F2FP, F2FP throughput drops; if independent, time is max(CVT, COMP).
//
// Packed-narrow F2FP peak on B300 = 32 /SM/clk (measured 31.7).
// So we use blk=512 × U=32 × N_CVT=16 × blocks=592 (4× SM) from the push sweep.
//
// N_CVT: number of independent F2FP round-trips per inner iter (each = 2 F2FPs)
// N_COMP: number of companion ops per inner iter
// COMP_TYPE: 0=FFMA 1=FMUL 2=IMAD 3=LOP3 (xor) 4=MUFU(ex2) 5=SHL 6=IADD3

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
    unsigned int p[N_CVT];
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

#if COMP_TYPE == 0 || COMP_TYPE == 1
    float c[N_COMP > 0 ? N_COMP : 1];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) c[m] = (float)(threadIdx.x + m + 1) * 1.0001f;
    const float ca = 1.0000001f;
#elif COMP_TYPE == 4  /* MUFU */
    float c[N_COMP > 0 ? N_COMP : 1];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) c[m] = (float)(threadIdx.x + m) * 0.01f + 1.0f;
#else
    unsigned int c[N_COMP > 0 ? N_COMP : 1];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) c[m] = threadIdx.x + m + 0xBEEF;
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            // Interleave CVTs and COMPs — the compiler is free to schedule
            // since each chain is independent from the others.
            #pragma unroll
            for (int k = 0; k < N_CVT; k++) {
                asm volatile(
                    "{ .reg .b16 _h;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                    : "+r"(p[k]));
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
    for (int k = 0; k < N_CVT; k++) acc ^= p[k];
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
