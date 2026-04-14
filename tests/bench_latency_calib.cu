// Latency calibration: same harness as bench_f2fp_depth, but with PIPE choosing
// the instruction type. If FFMA = 4 cy (documented Hopper/Blackwell) then any
// artifact in my harness should show up as deviation from 4.
//
// PIPE:
//   0 = FFMA (expected 4 cy) - CALIBRATION
//   1 = F2FP packed narrow round-trip (e4m3x2 ↔ f16x2)
//   2 = F2FP scalar f32→f16 (compiler folds reverse, so 1 F2FP + 2 LOP3 forced)
//   3 = F2F widening f16→f32 inside asm block (try to force non-fold)
//   4 = IMAD (expected 4 cy)
//   5 = FMUL (expected 4 cy)
//
// CHAIN_DEPTH: K ops in a single dependency chain per step. Step closes with 1 XOR (LOP3).

#ifndef PIPE
#define PIPE 1
#endif
#ifndef CHAIN_DEPTH
#define CHAIN_DEPTH 2
#endif
#ifndef N_CHAINS
#define N_CHAINS 1
#endif
#ifndef UNROLL
#define UNROLL 32
#endif
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 32
#endif
#ifndef MIN_BLOCKS
#define MIN_BLOCKS 1
#endif

#define _S(x) #x
#define S(x) _S(x)

// Stringify a chain of K dependent ops. We use 'do { K; } while(0)' pattern but
// expand manually to avoid C preprocessor headaches.

#if PIPE == 0   /* FFMA f32 */
  #define CHAIN_INIT float x = (float)(threadIdx.x + k) * 1.0001f;
  #define CHAIN_ONE(reg)  asm volatile("fma.rn.f32 %0, %0, %1, %2;" : "+f"(reg) : "f"(1.0000001f), "f"(0.0000001f));
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x1;" : "+r"(*(unsigned int*)&reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = __float_as_int(x);

#elif PIPE == 1  /* F2FP packed narrow round-trip */
  #define CHAIN_INIT unsigned int x = 0x3C003C01u ^ (threadIdx.x + k);
  #define CHAIN_ONE(reg) \
    asm volatile("{ .reg .b16 _h; cvt.rn.satfinite.e4m3x2.f16x2 _h, %0; cvt.rn.f16x2.e4m3x2 %0, _h; }" : "+r"(reg));
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x10001;" : "+r"(reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = x;
  // Note: each CHAIN_ONE is actually 2 F2FPs (forward + reverse).

#elif PIPE == 2  /* Scalar f32 <-> f16 (with LOP3 interposed to defeat compiler fold) */
  #define CHAIN_INIT float x = (float)(threadIdx.x + k) * 0.001f + 1.0f; unsigned short h;
  #define CHAIN_ONE(reg) \
    asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(h) : "f"(reg)); \
    asm volatile("xor.b16 %0, %0, 1;" : "+h"(h)); \
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(reg) : "h"(h));
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x10000;" : "+r"(*(unsigned int*)&reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = __float_as_int(x);

#elif PIPE == 3  /* F2F widen — f16→f32 via cvt (may compile to MOV/PRMT or F2F) */
  #define CHAIN_INIT unsigned short h = 0x3C00 ^ (threadIdx.x + k); float x;
  #define CHAIN_ONE(reg_float) \
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(reg_float) : "h"(h)); \
    asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(h) : "f"(reg_float));
  #define CHAIN_CLOSE(reg_float) asm volatile("xor.b16 %0, %0, 1;" : "+h"(h));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = (unsigned int)h;

#elif PIPE == 4  /* IMAD (integer) */
  #define CHAIN_INIT unsigned int x = threadIdx.x + k + 0x12345;
  #define CHAIN_ONE(reg) asm volatile("mad.lo.u32 %0, %0, 3, 1;" : "+r"(reg));
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x1;" : "+r"(reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = x;

#elif PIPE == 5  /* FMUL f32 */
  #define CHAIN_INIT float x = (float)(threadIdx.x + k + 1) * 1.00001f;
  #define CHAIN_ONE(reg) asm volatile("mul.rn.f32 %0, %0, %1;" : "+f"(reg) : "f"(1.0000001f));
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x1;" : "+r"(*(unsigned int*)&reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = __float_as_int(x);
#elif PIPE == 6  /* cvt.rna.tf32.f32 (potential F2F pipe) — round-trip via .b32 identity */
  #define CHAIN_INIT float x = (float)(threadIdx.x + k + 1) * 1.00001f;
  #define CHAIN_ONE(reg) { \
    unsigned int _u; \
    asm volatile("cvt.rna.satfinite.tf32.f32 %0, %1;" : "=r"(_u) : "f"(reg)); \
    asm volatile("mov.b32 %0, %1;" : "=f"(reg) : "r"(_u)); }
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x1;" : "+r"(*(unsigned int*)&reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = __float_as_int(x);
#elif PIPE == 7  /* cvt.rn.tf32.f32 (faster, used for tensor core setup) */
  #define CHAIN_INIT float x = (float)(threadIdx.x + k + 1) * 1.00001f;
  #define CHAIN_ONE(reg) { \
    unsigned int _u; \
    asm volatile("cvt.rn.satfinite.tf32.f32 %0, %1;" : "=r"(_u) : "f"(reg)); \
    asm volatile("mov.b32 %0, %1;" : "=f"(reg) : "r"(_u)); }
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x1;" : "+r"(*(unsigned int*)&reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = __float_as_int(x);
#elif PIPE == 8  /* cvt.rn.f16.f32 (NON-satfinite, may use F2F pipe) */
  #define CHAIN_INIT float x = (float)(threadIdx.x + k + 1) * 1.00001f; unsigned short h;
  #define CHAIN_ONE(reg) { \
    asm volatile("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(reg)); \
    asm volatile("xor.b16 %0, %0, 1;" : "+h"(h)); \
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(reg) : "h"(h)); }
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x10000;" : "+r"(*(unsigned int*)&reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = __float_as_int(x);
#elif PIPE == 9  /* cvt.rn.bf16.f32 (non-satfinite bf16) */
  #define CHAIN_INIT float x = (float)(threadIdx.x + k + 1) * 1.00001f; unsigned short b;
  #define CHAIN_ONE(reg) { \
    asm volatile("cvt.rn.bf16.f32 %0, %1;" : "=h"(b) : "f"(reg)); \
    asm volatile("xor.b16 %0, %0, 1;" : "+h"(b)); \
    asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(reg) : "h"(b)); }
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x10000;" : "+r"(*(unsigned int*)&reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = __float_as_int(x);
#elif PIPE == 10  /* cvt.rn.f16x2.f32  (packed, may be F2FP or F2F) */
  #define CHAIN_INIT float x = (float)(threadIdx.x + k + 1) * 1.00001f; unsigned int ph;
  #define CHAIN_ONE(reg) { \
    asm volatile("cvt.rn.f16x2.f32 %0, %1, %1;" : "=r"(ph) : "f"(reg)); \
    asm volatile("xor.b32 %0, %0, 0x10001;" : "+r"(ph)); \
    asm volatile("{ .reg .b16 lo, hi; mov.b32 {lo, hi}, %1; cvt.f32.f16 %0, lo; }" : "=f"(reg) : "r"(ph)); }
  #define CHAIN_CLOSE(reg) asm volatile("xor.b32 %0, %0, 0x10000;" : "+r"(*(unsigned int*)&reg));
  #define CHAIN_VAR x
  #define CHAIN_SINK unsigned int acc_k = __float_as_int(x);
#endif

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
    unsigned int acc_total = 0;

    #pragma unroll 1
    for (int k = 0; k < N_CHAINS; k++) {
        CHAIN_INIT;

        #pragma unroll 1
        for (int i = 0; i < ITERS; i += UNROLL) {
            #pragma unroll
            for (int j = 0; j < UNROLL; j++) {
                #if CHAIN_DEPTH >= 1
                  CHAIN_ONE(CHAIN_VAR);
                #endif
                #if CHAIN_DEPTH >= 2
                  CHAIN_ONE(CHAIN_VAR);
                #endif
                #if CHAIN_DEPTH >= 3
                  CHAIN_ONE(CHAIN_VAR);
                #endif
                #if CHAIN_DEPTH >= 4
                  CHAIN_ONE(CHAIN_VAR);
                #endif
                #if CHAIN_DEPTH >= 5
                  CHAIN_ONE(CHAIN_VAR);
                #endif
                #if CHAIN_DEPTH >= 6
                  CHAIN_ONE(CHAIN_VAR);
                #endif
                #if CHAIN_DEPTH >= 7
                  CHAIN_ONE(CHAIN_VAR);
                #endif
                #if CHAIN_DEPTH >= 8
                  CHAIN_ONE(CHAIN_VAR);
                #endif
                CHAIN_CLOSE(CHAIN_VAR);
            }
        }
        CHAIN_SINK;
        acc_total ^= acc_k;
    }

    if ((int)acc_total == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc_total;
}
