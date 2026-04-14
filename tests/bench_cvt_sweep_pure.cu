// CVT sweep — measures each variant's ALU-pipe peak with minimal feedback.
// UNPACK variants: feedback via low-bit register alias (no LOP3).
// PACK variants: feedback via zero-ext cast (one LOP3 per iter, ~32/clk).

#ifndef N_CHAINS
#define N_CHAINS 8
#endif
#ifndef UNROLL
#define UNROLL 16
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif
#ifndef OP
#define OP 0
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned int v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++)
        v[k] = 0x3C003C01u ^ (threadIdx.x * 137 + k * 23);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned int tmp;
                unsigned short stmp;

#if OP == 0
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)v[k]));
                v[k] = tmp;
#elif OP == 1
                asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)v[k]));
                v[k] = tmp;
#elif OP == 2
                // FP4 unpack: input is .b8, take low byte from u16
                asm volatile("{ .reg .b8 _b, _pad; mov.b16 {_b,_pad}, %1; cvt.rn.f16x2.e2m1x2 %0, _b; }"
                             : "=r"(tmp) : "h"((unsigned short)v[k]));
                v[k] = tmp;
#elif OP == 3
                asm volatile("cvt.rn.f16x2.e2m3x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)v[k]));
                v[k] = tmp;
#elif OP == 4
                asm volatile("cvt.rn.f16x2.e3m2x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)v[k]));
                v[k] = tmp;
#elif OP == 5
                asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(tmp) : "h"((unsigned short)v[k]));
                v[k] = tmp;
// ==== PACK variants ====
#elif OP == 10
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(stmp) : "r"(v[k]));
                v[k] = (unsigned int)stmp;
#elif OP == 11
                asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(stmp) : "r"(v[k]));
                v[k] = (unsigned int)stmp;
#elif OP == 12
                // FP4 pack: output is .b8, wrap in register-level mov
                asm volatile("{ .reg .b8 _b; cvt.rn.satfinite.e2m1x2.f16x2 _b, %1; mov.b16 %0,{_b,_b}; }"
                             : "=h"(stmp) : "r"(v[k]));
                v[k] = (unsigned int)stmp;
#elif OP == 13
                {
                    float fa = __int_as_float(v[k]);
                    float fb = __int_as_float(v[k] ^ 0xABCDu);
                    asm volatile("cvt.rn.satfinite.e2m3x2.f32 %0, %2, %1;" : "=h"(stmp) : "f"(fa), "f"(fb));
                    v[k] = (unsigned int)stmp;
                }
#elif OP == 14
                {
                    float fa = __int_as_float(v[k]);
                    float fb = __int_as_float(v[k] ^ 0xABCDu);
                    asm volatile("cvt.rn.satfinite.e3m2x2.f32 %0, %2, %1;" : "=h"(stmp) : "f"(fa), "f"(fb));
                    v[k] = (unsigned int)stmp;
                }
#elif OP == 15
                {
                    float fa = __int_as_float(v[k]);
                    float fb = __int_as_float(v[k] ^ 0xABCDu);
                    asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(stmp) : "f"(fa), "f"(fb));
                    v[k] = (unsigned int)stmp;
                }
#elif OP == 16
                {
                    float fa = __int_as_float(v[k]);
                    float fb = __int_as_float(v[k] ^ 0xABCDu);
                    asm volatile("{ .reg .b8 _b; cvt.rn.satfinite.e2m1x2.f32 _b, %2, %1; mov.b16 %0,{_b,_b}; }"
                                 : "=h"(stmp) : "f"(fa), "f"(fb));
                    v[k] = (unsigned int)stmp;
                }
#endif
                (void)stmp; (void)tmp;
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
