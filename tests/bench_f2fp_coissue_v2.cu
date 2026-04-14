// F2FP + companion co-issue, v2
// Use PACK (single-slot, 32/SM/clk) to leave 1 SFU slot "free" per cycle.
// If companion hits the SFU, F2FP throughput drops.
// If companion is on a different pipe, F2FP throughput unchanged (at any N_COMP).
//
// F2FP used: cvt.rn.satfinite.e4m3x2.f16x2 only (PACK, MERGE_C, 32/SM/clk = 1 SFU slot)
//
// COMP_TYPE:
//  0 = FFMA     (FMA pipe)
//  1 = FMUL     (FMA pipe)
//  2 = IMAD     (INT pipe)
//  3 = LOP3     (INT pipe)
//  4 = MUFU.ex2 (SFU pipe - candidate)
//  5 = MUFU.rsqrt (SFU - 2 slot)
//  6 = F2F.f16.f32 (F2F pipe, reported separate from F2FP)
//  7 = HFMA2   (half FMA pipe)
//  8 = HADD2   (half add)
//  9 = I2F     (int-to-float)
// 10 = F2I     (float-to-int)
// 11 = SHFL    (warp shuffle)
// 12 = POPC    (integer popc)
// 13 = TANH.approx.f16 (native half-precision tanh - goes through SFU)
// 14 = IADD3   (int pipe add3)
// 15 = SEL     (select)
// 16 = VOTE    (warp vote)

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
#ifndef COMP_TYPE
#define COMP_TYPE 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, 1)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    // F2FP chains (unpack + pack round-trip, 16 bit input / output)
    unsigned int p[N_CVT];
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

    // Companion chains (typed by COMP_TYPE)
#if COMP_TYPE == 0 || COMP_TYPE == 1 || COMP_TYPE == 4 || COMP_TYPE == 5 || COMP_TYPE == 6 || COMP_TYPE == 9 || COMP_TYPE == 10 || COMP_TYPE == 13
    float cf[N_COMP > 0 ? N_COMP : 1];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) cf[m] = (float)(threadIdx.x + m + 1) * 1.0001f;
    const float ca = 1.0000001f;
#endif
#if COMP_TYPE == 7 || COMP_TYPE == 8
    unsigned int ch[N_COMP > 0 ? N_COMP : 1];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) ch[m] = 0x3C003C01u ^ (threadIdx.x + m + 7);
#endif
#if COMP_TYPE == 2 || COMP_TYPE == 3 || COMP_TYPE == 11 || COMP_TYPE == 12 || COMP_TYPE == 14 || COMP_TYPE == 15 || COMP_TYPE == 16
    unsigned int ci[N_COMP > 0 ? N_COMP : 1];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) ci[m] = threadIdx.x + m + 0xBEEF;
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CVT; k++) {
                // Do 2× pack + unpack round-trip to reach 2 SFU-slots/cycle saturated
                // (same as bench_f2fp_coissue.cu). This lets us detect sharing.
                asm volatile(
                    "{ .reg .b16 _h;\n\t"
                    "  cvt.rn.satfinite.e4m3x2.f16x2 _h, %0;\n\t"
                    "  cvt.rn.f16x2.e4m3x2 %0, _h; }"
                    : "+r"(p[k]));
            }
            #pragma unroll
            for (int m = 0; m < N_COMP; m++) {
#if COMP_TYPE == 0
                asm volatile("fma.rn.f32 %0, %0, %1, %1;" : "+f"(cf[m]) : "f"(ca));
#elif COMP_TYPE == 1
                asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(cf[m]) : "f"(ca));
#elif COMP_TYPE == 2
                asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(ci[m]));
#elif COMP_TYPE == 3
                asm volatile("xor.b32 %0, %0, 0x1;" : "+r"(ci[m]));
#elif COMP_TYPE == 4
                asm volatile("ex2.approx.f32 %0, %0;" : "+f"(cf[m]));
#elif COMP_TYPE == 5
                asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(cf[m]));
#elif COMP_TYPE == 6
                // Single F2F half-round-trip: f32->f16 then back. Each m contributes 2 F2F ops.
                asm volatile("{ .reg .b16 h; cvt.rn.f16.f32 h, %0; cvt.f32.f16 %0, h; }" : "+f"(cf[m]));
#elif COMP_TYPE == 17
                // Single F2F only: use a temporary destination, read back via mov
                asm volatile("{ .reg .b16 h; .reg .f32 t; cvt.rn.f16.f32 h, %0; mov.f32 t, %0; cvt.f32.f16 %0, h; add.f32 %0, %0, t; }" : "+f"(cf[m]));
#elif COMP_TYPE == 7
                // HFMA2 half FMA
                asm volatile("fma.rn.f16x2 %0, %0, %0, %0;" : "+r"(ch[m]));
#elif COMP_TYPE == 8
                // HADD2
                asm volatile("add.rn.f16x2 %0, %0, %0;" : "+r"(ch[m]));
#elif COMP_TYPE == 9
                // I2F (int -> f32) via cvt
                asm volatile("{ .reg .s32 t; cvt.rzi.s32.f32 t, %0; cvt.rn.f32.s32 %0, t; }" : "+f"(cf[m]));
#elif COMP_TYPE == 10
                // F2I
                asm volatile("{ .reg .s32 t; cvt.rzi.s32.f32 t, %0; cvt.rn.f32.s32 %0, t; }" : "+f"(cf[m]));
#elif COMP_TYPE == 11
                // SHFL (warp shuffle down, wraps)
                asm volatile("shfl.sync.bfly.b32 %0, %0, 1, 0x1f, 0xffffffff;" : "+r"(ci[m]));
#elif COMP_TYPE == 12
                asm volatile("popc.b32 %0, %0;" : "+r"(ci[m]));
#elif COMP_TYPE == 13
                asm volatile("tanh.approx.f32 %0, %0;" : "+f"(cf[m]));
#elif COMP_TYPE == 14
                asm volatile("add.u32 %0, %0, 1;" : "+r"(ci[m]));
#elif COMP_TYPE == 15
                asm volatile("{ .reg .pred p; setp.lt.u32 p, %0, 16; selp.b32 %0, %0, 0, p; }" : "+r"(ci[m]));
#elif COMP_TYPE == 16
                {
                unsigned int v;
                asm volatile("vote.sync.ballot.b32 %0, 1, 0xffffffff;" : "=r"(v));
                ci[m] ^= v;
                }
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CVT; k++) acc ^= p[k];
    #pragma unroll
    for (int m = 0; m < N_COMP; m++) {
#if COMP_TYPE == 0 || COMP_TYPE == 1 || COMP_TYPE == 4 || COMP_TYPE == 5 || COMP_TYPE == 6 || COMP_TYPE == 9 || COMP_TYPE == 10 || COMP_TYPE == 13
        acc ^= __float_as_int(cf[m]);
#elif COMP_TYPE == 7 || COMP_TYPE == 8
        acc ^= ch[m];
#else
        acc ^= ci[m];
#endif
    }
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
