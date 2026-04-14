// Paired-op benchmark with DCE-proof feedback chains.
// Runs N_A ops of TYPE_A + N_B ops of TYPE_B per iter (or either alone).
// Each type uses its own feedback chain to prevent compiler folding.
// Ops available:
//   0 = F2FP.unpack (f16x2 ← e4m3x2)
//   1 = F2FP.pack   (e4m3x2 ← f16x2 with MERGE_C)
//   2 = FFMA.f32
//   3 = FMUL.f32
//   4 = IADD (add.u32 x, x, 1)  -- self-increment (ptxas can't fold)
//   5 = LOP3.xor (xor with thread-const)
//   6 = MUFU.EX2
//   7 = MUFU.RSQ
//   8 = HFMA2 (f16x2 fma self)
//   9 = IMAD (self-multiply-add)

#ifndef TYPE_A
#define TYPE_A 0
#endif
#ifndef N_A
#define N_A 4
#endif
#ifndef TYPE_B
#define TYPE_B 0
#endif
#ifndef N_B
#define N_B 0
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 512
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 2
#endif

// For each TYPE, define:
// INIT(vec, k) : initial value
// EMIT(vec) : one op per CHAIN
// SINK(vec, acc) : contribute to final accumulator

#define INIT_F2FP_UNPACK(u16v, k) u16v = 0x3C01 ^ (threadIdx.x + k)
#define EMIT_F2FP_UNPACK(u16v, tmp32) \
    asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(tmp32) : "h"(u16v)); \
    u16v = (unsigned short)tmp32

#define INIT_F2FP_PACK(u32v, k) u32v = 0x3C003C01u ^ (threadIdx.x + k)
#define EMIT_F2FP_PACK(u32v, tmp16) \
    asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(tmp16) : "r"(u32v)); \
    u32v = (unsigned int)tmp16

#define INIT_FFMA(fv, k) fv = 1.00001f + 0.0001f * (threadIdx.x + k)
#define EMIT_FFMA(fv) asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(1.000001f), "f"(0.9999f))

#define INIT_FMUL(fv, k) fv = 1.00001f + 0.0001f * (threadIdx.x + k)
#define EMIT_FMUL(fv) asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(1.0000001f))

#define INIT_IADD(uv, k) uv = 0xBEEF0000u + (threadIdx.x + k)
#define EMIT_IADD(uv) asm volatile("add.u32 %0, %0, 1;" : "+r"(uv))

#define INIT_LOP3(uv, k) uv = 0xDEAD0000u + (threadIdx.x + k)
#define EMIT_LOP3(uv) asm volatile("xor.b32 %0, %0, 0xAAAAAAAB;" : "+r"(uv))

#define INIT_EX2(fv, k) fv = 1.0001f + 0.0001f * (threadIdx.x + k)
#define EMIT_EX2(fv) asm volatile("ex2.approx.f32 %0, %0;" : "+f"(fv))

#define INIT_RSQ(fv, k) fv = 1.0001f + 0.0001f * (threadIdx.x + k)
#define EMIT_RSQ(fv) asm volatile("rsqrt.approx.f32 %0, %0;" : "+f"(fv))

#define INIT_HFMA2(uv, k) uv = 0x3C003C01u ^ (threadIdx.x + k)
#define EMIT_HFMA2(uv) asm volatile("fma.rn.f16x2 %0, %0, %1, %1;" : "+r"(uv) : "r"((unsigned int)0x3C003C00u))

#define INIT_IMAD(uv, k) uv = (threadIdx.x + k + 1)
#define EMIT_IMAD(uv) asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(uv))

// Helper to instantiate a block based on TYPE
#define TYPE_STATE(T, vec_name, k) \
    _Pragma("GCC diagnostic push") \
    _Pragma("GCC diagnostic ignored \"-Wunused-variable\"")

// Per-type storage — we'll use a union approach: pick storage per TYPE
// To keep it simple, we manually instantiate by TYPE_A/TYPE_B.
// We'll use 3 storage slots: s_f2fp_u16[], s_u32[], s_f32[] that are reused.

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    // TYPE_A state
    unsigned short a_h[N_A];
    unsigned int   a_u[N_A];
    float          a_f[N_A];
    #pragma unroll
    for (int k = 0; k < N_A; k++) {
#if TYPE_A == 0
        INIT_F2FP_UNPACK(a_h[k], k);
#elif TYPE_A == 1
        INIT_F2FP_PACK(a_u[k], k);
#elif TYPE_A == 2
        INIT_FFMA(a_f[k], k);
#elif TYPE_A == 3
        INIT_FMUL(a_f[k], k);
#elif TYPE_A == 4
        INIT_IADD(a_u[k], k);
#elif TYPE_A == 5
        INIT_LOP3(a_u[k], k);
#elif TYPE_A == 6
        INIT_EX2(a_f[k], k);
#elif TYPE_A == 7
        INIT_RSQ(a_f[k], k);
#elif TYPE_A == 8
        INIT_HFMA2(a_u[k], k);
#elif TYPE_A == 9
        INIT_IMAD(a_u[k], k);
#endif
    }

#if N_B > 0
    unsigned short b_h[N_B];
    unsigned int   b_u[N_B];
    float          b_f[N_B];
    #pragma unroll
    for (int k = 0; k < N_B; k++) {
  #if TYPE_B == 0
        INIT_F2FP_UNPACK(b_h[k], k + 100);
  #elif TYPE_B == 1
        INIT_F2FP_PACK(b_u[k], k + 100);
  #elif TYPE_B == 2
        INIT_FFMA(b_f[k], k + 100);
  #elif TYPE_B == 3
        INIT_FMUL(b_f[k], k + 100);
  #elif TYPE_B == 4
        INIT_IADD(b_u[k], k + 100);
  #elif TYPE_B == 5
        INIT_LOP3(b_u[k], k + 100);
  #elif TYPE_B == 6
        INIT_EX2(b_f[k], k + 100);
  #elif TYPE_B == 7
        INIT_RSQ(b_f[k], k + 100);
  #elif TYPE_B == 8
        INIT_HFMA2(b_u[k], k + 100);
  #elif TYPE_B == 9
        INIT_IMAD(b_u[k], k + 100);
  #endif
    }
#endif

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_A; k++) {
                unsigned int tmp32;
                unsigned short tmp16;
#if TYPE_A == 0
                EMIT_F2FP_UNPACK(a_h[k], tmp32);
#elif TYPE_A == 1
                EMIT_F2FP_PACK(a_u[k], tmp16);
#elif TYPE_A == 2
                EMIT_FFMA(a_f[k]);
#elif TYPE_A == 3
                EMIT_FMUL(a_f[k]);
#elif TYPE_A == 4
                EMIT_IADD(a_u[k]);
#elif TYPE_A == 5
                EMIT_LOP3(a_u[k]);
#elif TYPE_A == 6
                EMIT_EX2(a_f[k]);
#elif TYPE_A == 7
                EMIT_RSQ(a_f[k]);
#elif TYPE_A == 8
                EMIT_HFMA2(a_u[k]);
#elif TYPE_A == 9
                EMIT_IMAD(a_u[k]);
#endif
            }
#if N_B > 0
            #pragma unroll
            for (int k = 0; k < N_B; k++) {
                unsigned int tmp32;
                unsigned short tmp16;
  #if TYPE_B == 0
                EMIT_F2FP_UNPACK(b_h[k], tmp32);
  #elif TYPE_B == 1
                EMIT_F2FP_PACK(b_u[k], tmp16);
  #elif TYPE_B == 2
                EMIT_FFMA(b_f[k]);
  #elif TYPE_B == 3
                EMIT_FMUL(b_f[k]);
  #elif TYPE_B == 4
                EMIT_IADD(b_u[k]);
  #elif TYPE_B == 5
                EMIT_LOP3(b_u[k]);
  #elif TYPE_B == 6
                EMIT_EX2(b_f[k]);
  #elif TYPE_B == 7
                EMIT_RSQ(b_f[k]);
  #elif TYPE_B == 8
                EMIT_HFMA2(b_u[k]);
  #elif TYPE_B == 9
                EMIT_IMAD(b_u[k]);
  #endif
            }
#endif
        }
    }

    // Sink all live state
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_A; k++) {
#if TYPE_A == 0 || TYPE_A == 1 || TYPE_A == 4 || TYPE_A == 5 || TYPE_A == 8 || TYPE_A == 9
        acc ^= (unsigned int)a_u[k] ^ (unsigned int)a_h[k];
#else
        acc ^= __float_as_int(a_f[k]);
#endif
    }
#if N_B > 0
    #pragma unroll
    for (int k = 0; k < N_B; k++) {
  #if TYPE_B == 0 || TYPE_B == 1 || TYPE_B == 4 || TYPE_B == 5 || TYPE_B == 8 || TYPE_B == 9
        acc ^= (unsigned int)b_u[k] ^ (unsigned int)b_h[k];
  #else
        acc ^= __float_as_int(b_f[k]);
  #endif
    }
#endif
    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
