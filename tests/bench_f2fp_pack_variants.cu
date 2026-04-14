// Test different PACK variants to see which hit 64/SM/clk vs capped at 32.
//
// VARIANT:
//   0 = cvt.rn.satfinite.e4m3x2.f16x2  (UNPACK_B_MERGE_C)  — the 32/clk case
//   1 = cvt.rn.f16x2.e4m3x2            (UNPACK_B, no MERGE_C) — 64/clk
//   2 = cvt.rn.f16x2.f32 (pair)        (PACK_AB, no MERGE_C) — hypothesis: 64/clk
//   3 = cvt.rn.bf16x2.f32 (pair)       (PACK_AB, no MERGE_C) — same hypothesis
//   4 = cvt.rn.satfinite.tf32.f32      (PACK_B, no MERGE_C) — tf32
//   5 = cvt.rn.satfinite.e4m3x2.f32    (PACK_AB_MERGE_C) — f32→narrow pack (should be 32)
//   6 = cvt.rn.satfinite.f16.f32       (scalar MERGE_C) — 24/clk scalar

#ifndef VARIANT
#define VARIANT 0
#endif
#ifndef N_CHAINS
#define N_CHAINS 16
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

extern "C" __global__ __launch_bounds__(BLOCK_SIZE, MIN_BLOCKS)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int unused_2) {
#if VARIANT == 0 || VARIANT == 1 || VARIANT == 2 || VARIANT == 3 || VARIANT == 4
    // These use u32 chain for input and produce u32 or u16 output
    unsigned int p[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) p[k] = 0x3C003C01u ^ (threadIdx.x + k);

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
#if VARIANT == 0
                // f16x2 → e4m3x2 (PACK via UNPACK_B_MERGE_C, capped at 32)
                unsigned short h;
                asm volatile("cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;" : "=h"(h) : "r"(p[k]));
                p[k] = (unsigned int)h;
#elif VARIANT == 1
                // e4m3x2 → f16x2 (UNPACK, pure, 64/clk)
                unsigned short hin = (unsigned short)p[k];
                asm volatile("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(p[k]) : "h"(hin));
#elif VARIANT == 2
                // f32 pair → f16x2 (PACK_AB, NO MERGE_C — output is full 32-bit)
                // We use p[k] as source (reinterpret as two f32s via split+widen,
                // keeping minimal feedback overhead)
                float f_lo = __int_as_float(p[k]);
                float f_hi = __int_as_float(p[k] ^ 0x00010000u);
                asm volatile("cvt.rn.f16x2.f32 %0, %2, %1;" : "=r"(p[k]) : "f"(f_lo), "f"(f_hi));
#elif VARIANT == 3
                // f32 pair → bf16x2 (PACK_AB)
                float f_lo = __int_as_float(p[k]);
                float f_hi = __int_as_float(p[k] ^ 0x00010000u);
                asm volatile("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(p[k]) : "f"(f_lo), "f"(f_hi));
#elif VARIANT == 4
                // f32 → tf32 (PACK_B, single source)
                float f = __int_as_float(p[k]);
                asm volatile("cvt.rn.satfinite.tf32.f32 %0, %1;" : "=r"(p[k]) : "f"(f));
#endif
            }
        }
    }

    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= p[k];

#elif VARIANT == 5
    // f32 pair → e4m3x2 (PACK_AB_MERGE_C — pack narrow from f32)
    float r[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) r[k] = (float)(threadIdx.x + k) * 1.0001f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short h;
                asm volatile("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;" : "=h"(h) : "f"(r[k]), "f"(r[k]+0.5f));
                r[k] = __int_as_float((unsigned int)h);
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= __float_as_int(r[k]);

#elif VARIANT == 6
    // f32 → f16 (scalar MERGE_C, measured ~21-24/clk earlier)
    float r[N_CHAINS];
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) r[k] = (float)(threadIdx.x + k) * 1.0001f;

    #pragma unroll 1
    for (int i = 0; i < ITERS; i += UNROLL) {
        #pragma unroll
        for (int j = 0; j < UNROLL; j++) {
            #pragma unroll
            for (int k = 0; k < N_CHAINS; k++) {
                unsigned short h;
                asm volatile("cvt.rn.satfinite.f16.f32 %0, %1;" : "=h"(h) : "f"(r[k]));
                r[k] = __int_as_float((unsigned int)h);
            }
        }
    }
    unsigned int acc = 0;
    #pragma unroll
    for (int k = 0; k < N_CHAINS; k++) acc ^= __float_as_int(r[k]);
#endif

    if ((int)acc == seed) ((unsigned int*)C)[blockIdx.x * blockDim.x + threadIdx.x] = acc;
}
