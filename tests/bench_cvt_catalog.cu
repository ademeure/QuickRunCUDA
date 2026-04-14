// Exhaustive CVT catalog. One OP per variant; store in u64 for flexibility.

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
    unsigned long long v[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++)
        v[k] = 0x3C003C013F8000ULL ^ (threadIdx.x * 0x100000007ULL + k * 17);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned long long w = v[k];
                unsigned int lo = (unsigned int)w;
                unsigned short s  = (unsigned short)w;

// ==== FP32 → narrow (PACK from scalar/pair) ====
#if OP == 0   // f32 pair → e4m3x2 (FP8)
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(s)
                             : "f"(__int_as_float(lo)), "f"(__int_as_float(lo^0xAB)));
#elif OP == 1 // f32 pair → e5m2x2 (FP8)
                asm volatile("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;" : "=h"(s)
                             : "f"(__int_as_float(lo)), "f"(__int_as_float(lo^0xAB)));
#elif OP == 2 // f32 pair → e2m3x2 (FP6)
                asm volatile("cvt.rn.satfinite.e2m3x2.f32 %0, %2, %1;" : "=h"(s)
                             : "f"(__int_as_float(lo)), "f"(__int_as_float(lo^0xAB)));
#elif OP == 3 // f32 pair → e3m2x2 (FP6)
                asm volatile("cvt.rn.satfinite.e3m2x2.f32 %0, %2, %1;" : "=h"(s)
                             : "f"(__int_as_float(lo)), "f"(__int_as_float(lo^0xAB)));
#elif OP == 4 // f32 pair → e2m1x2 (FP4, needs b8 wrap)
                asm volatile("{.reg .b8 _b; cvt.rn.satfinite.e2m1x2.f32 _b, %2, %1; mov.b16 %0,{_b,_b};}"
                             : "=h"(s) : "f"(__int_as_float(lo)), "f"(__int_as_float(lo^0xAB)));
#elif OP == 5 // f32 → f16
                asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(s) : "f"(__int_as_float(lo)));
#elif OP == 6 // f32 → bf16
                asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(s) : "f"(__int_as_float(lo)));
#elif OP == 7 // f32 → ue8m0 (block scale; sat finite)
                asm volatile("cvt.rp.satfinite.ue8m0x2.f32 %0, %2, %1;" : "=h"(s)
                             : "f"(__int_as_float(lo)), "f"(__int_as_float(lo^0xAB)));
// ==== f16x2 → narrow (PACK from packed halves) ====
#elif OP == 10 // f16x2 → e4m3x2
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(s) : "r"(lo));
#elif OP == 11 // f16x2 → e5m2x2
                asm volatile("cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;" : "=h"(s) : "r"(lo));
#elif OP == 12 // f16x2 → e2m1x2 (FP4)
                asm volatile("{.reg .b8 _b; cvt.rn.satfinite.e2m1x2.f16x2 _b, %1; mov.b16 %0,{_b,_b};}"
                             : "=h"(s) : "r"(lo));
// ==== narrow → f16x2 (UNPACK) ====
#elif OP == 20 // e4m3x2 → f16x2
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo) : "h"(s));
#elif OP == 21 // e5m2x2 → f16x2
                asm volatile("cvt.rn.f16x2.e5m2x2 %0, %1;" : "=r"(lo) : "h"(s));
#elif OP == 22 // e2m1x2 → f16x2 (FP4 input: b8 wrap)
                asm volatile("{.reg .b8 _b,_p; mov.b16 {_b,_p}, %1; cvt.rn.f16x2.e2m1x2 %0, _b;}"
                             : "=r"(lo) : "h"(s));
#elif OP == 23 // e2m3x2 → f16x2 (FP6)
                asm volatile("cvt.rn.f16x2.e2m3x2 %0, %1;" : "=r"(lo) : "h"(s));
#elif OP == 24 // e3m2x2 → f16x2 (FP6)
                asm volatile("cvt.rn.f16x2.e3m2x2 %0, %1;" : "=r"(lo) : "h"(s));
#elif OP == 25 // ue8m0x2 → bf16x2
                asm volatile("cvt.rn.bf16x2.ue8m0x2 %0, %1;" : "=r"(lo) : "h"(s));
// ==== FP wide conversions ====
#elif OP == 30 // f16 → f32
                {float f; asm volatile("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(s));
                 lo = __float_as_int(f);}
#elif OP == 31 // bf16 → f32
                {float f; asm volatile("cvt.f32.bf16 %0, %1;" : "=f"(f) : "h"(s));
                 lo = __float_as_int(f);}
#elif OP == 32 // f16 → bf16
                asm volatile("{.reg .f32 t; cvt.f32.f16 t, %1; cvt.rn.bf16.f32 %0, t;}" : "=h"(s) : "h"(s));
#elif OP == 33 // f16x2 unpack to 2 f32s
                {float f0, f1; asm volatile("{.reg .f16 a,b; mov.b32 {a,b}, %2; cvt.f32.f16 %0, a; cvt.f32.f16 %1, b;}"
                    : "=f"(f0), "=f"(f1) : "r"(lo));
                 lo = __float_as_int(f0) ^ __float_as_int(f1);}
// ==== INT ↔ FP ====
#elif OP == 40 // f32 → s32 rni
                {int ix; asm volatile("cvt.rni.s32.f32 %0, %1;" : "=r"(ix) : "f"(__int_as_float(lo)));
                 lo = (unsigned)ix;}
#elif OP == 41 // f32 → u32 rni
                asm volatile("cvt.rni.u32.f32 %0, %1;" : "=r"(lo) : "f"(__int_as_float(lo)));
#elif OP == 42 // f32 → s64
                {long long ix; asm volatile("cvt.rni.s64.f32 %0, %1;" : "=l"(ix) : "f"(__int_as_float(lo)));
                 w = (unsigned long long)ix;}
#elif OP == 43 // f32 → u8 sat
                asm volatile("cvt.rni.sat.u8.f32 %0, %1;" : "+r"(lo) : "f"(__int_as_float(lo)));
#elif OP == 44 // f32 → s8 sat
                asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "+r"(lo) : "f"(__int_as_float(lo)));
#elif OP == 45 // f32 → f32 rz (round-to-zero "self" cvt)
                {float f; asm volatile("cvt.rz.f32.f32 %0, %1;" : "=f"(f) : "f"(__int_as_float(lo)));
                 lo = __float_as_int(f);}
#elif OP == 46 // s32 → f32
                {float f; asm volatile("cvt.rn.f32.s32 %0, %1;" : "=f"(f) : "r"((int)lo));
                 lo = __float_as_int(f);}
#elif OP == 47 // u32 → f32
                {float f; asm volatile("cvt.rn.f32.u32 %0, %1;" : "=f"(f) : "r"(lo));
                 lo = __float_as_int(f);}
#elif OP == 48 // s64 → f32
                {float f; asm volatile("cvt.rn.f32.s64 %0, %1;" : "=f"(f) : "l"(w));
                 lo = __float_as_int(f);}
#elif OP == 49 // u8 → u32 (zero-ext)
                asm volatile("cvt.u32.u8 %0, %1;" : "=r"(lo) : "r"((unsigned char)lo));
// ==== INT ↔ INT ====
#elif OP == 50 // s32 → s16 (sat)
                {short sx; asm volatile("cvt.sat.s16.s32 %0, %1;" : "=h"(sx) : "r"((int)lo));
                 s = (unsigned short)sx;}
#elif OP == 51 // u32 → u16 (sat)
                asm volatile("cvt.sat.u16.u32 %0, %1;" : "=h"(s) : "r"(lo));
#elif OP == 52 // u64 → u32
                asm volatile("cvt.u32.u64 %0, %1;" : "=r"(lo) : "l"(w));
#elif OP == 53 // s32 → u8 (sat)
                {unsigned char u8; asm volatile("cvt.sat.u8.s32 %0, %1;" : "=r"(lo) : "r"((int)lo));}
#endif

                v[k] = (w & 0xFFFFFFFF00000000ULL) | lo;
                (void)s;
            }
        }
    }
    unsigned long long acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= v[k];
    if (acc == (unsigned long long)seed) ((unsigned long long*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
