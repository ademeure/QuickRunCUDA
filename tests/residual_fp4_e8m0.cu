// residual_fp4_e8m0.cu
//
// 1D BF16 or FP32 → "2× NVFP4" residual quantization with e8m0+e8m0 scales.
//
// Produces TWO FP4 outputs per input:
//   Primary:  e2m1 payload × e8m0 scale  (per-16-group)
//   Residual: e2m1 payload × e8m0 scale  (per-16-group)
// Final reconstruction: dq = dq_primary + dq_residual.
//
// Compile-time flags (-H):
//   INPUT_BF16        1 = BF16 input (default), 0 = FP32 input
//   NUM_CANDIDATES    1 or 2 (default 2, four/six)
//   STOCHASTIC_ROUND  0 = RTN (default), 1 = stochastic rounding
//                     SR uses low bits of neighbouring input bf16 values as
//                     random source (no RNG state).
//   ERROR_METRIC_MAE  0 = L2 winner (default, RMSE-optimal), 1 = L1 winner (MAE)
//   GROUPS_PER_THREAD 1, 2, or 4 (default 4 for load amortisation)
//
// Output layout per group (8+8+1+1 = 18 bytes):
//   [0..7]   primary FP4 bytes (8 e2m1x2 bytes)
//   [8..15]  residual FP4 bytes
//   [16]     e8m0 byte for primary scale
//   [17]     e8m0 byte for residual scale
//
// Args:
//   A: input, n_elems × (2 bytes bf16 or 4 bytes f32) as float* (NVRTC signature)
//   B: output bytes (use (unsigned char*)B)
//   C: per-tensor stats (C[0] = AMAX_CONST used, for diagnostics)
//   n_threads = number of groups × GROUPS_PER_THREAD  (actually n_threads = total threads, = n_groups/GPT)
//   cols_param, unused: ignored

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef INPUT_BF16
#define INPUT_BF16 1
#endif
#ifndef NUM_CANDIDATES
#define NUM_CANDIDATES 2
#endif
// Separate NCs per pass: residual quality barely benefits from NC=2 (residuals
// have narrow dynamic range after primary), but halves the residual-pass work.
#ifndef PRIMARY_NC
#define PRIMARY_NC NUM_CANDIDATES
#endif
#ifndef RESIDUAL_NC
#define RESIDUAL_NC 1
#endif
#ifndef STOCHASTIC_ROUND
#define STOCHASTIC_ROUND 0
#endif
#ifndef ERROR_METRIC_MAE
#define ERROR_METRIC_MAE 0
#endif
#ifndef GROUPS_PER_THREAD
#define GROUPS_PER_THREAD 4
#endif

#define VSIZE      16
#define FP4_BYTES   8
#define AMAX_CONST  1.3f
#define SCALE_OVERRIDE 1.0f

// ---------- helpers ----------

// bf16 extraction: <<16 = exact cast to f32 (bf16 is just f32 with low bits zeroed).
#define BF16_LO(w) __int_as_float((unsigned int)(w) << 16)
#define BF16_HI(w) __int_as_float((unsigned int)(w) & 0xFFFF0000u)

// Round |x| to nearest power of 2, return the f32 value (= descale).
// Expects x > 0. For x = 0 returns 2^-127 (avoids /0 later).
// e8m0 encoding: byte = (biased exponent), represents 2^(byte - 127).
static __device__ __forceinline__ unsigned char f32_to_e8m0_byte(float x, float& descale) {
    // Fastpath: extract exponent bits directly.
    unsigned int u = __float_as_uint(fmaxf(x, 1.175494e-38f));  // clamp to min normal
    // u = s.eeeeeeee.mmmmmmmmmmmmmmmmmmmmmmm
    unsigned int exp_bias = (u >> 23) & 0xFFu;  // raw biased exponent in [1, 254]
    unsigned int mant     = u & 0x7FFFFFu;
    // Round-half-to-even on the log: if top bit of mantissa is set AND the rest aren't all zero,
    // we round UP (to next power of 2). Actually: 2^e × 1.m ≥ 2^e × sqrt(2) ≈ 1.414
    // 1.m ≥ sqrt(2) iff m ≥ floor(sqrt(2)-1)×2^23 = 0x3504F3. Round up if mantissa > that threshold.
    // For simplicity, round up when top mantissa bit is set (≈ 1.m ≥ 1.5 → closer to 2 than 1).
    // This rounds in log-space but toward next power of 2 on ≥1.5, which is close enough.
    // For more accurate (nearest-in-log), use the 0x3504F3 threshold (~sqrt(2)).
    unsigned int next = (mant >= 0x3504F3u) ? 1u : 0u;
    unsigned int rounded_exp = exp_bias + next;
    if (rounded_exp > 254u) rounded_exp = 254u;  // saturate at 2^127
    descale = __uint_as_float(rounded_exp << 23);
    return (unsigned char)rounded_exp;
}

static __device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

// Fused quant+dequant for FP4 (e2m1x2) — keeps .b8 inside asm so no byte-mask LOP3 inserted.
// nearest-rounding (RTN).
static __device__ __forceinline__ void quant_dequant_bf16x2(
    unsigned int bf16x2_in,
    unsigned short& out_byte,       // e2m1x2 byte in low 8 bits
    unsigned int& out_dq_bf16x2)    // dequant bf16x2
{
    asm("{ .reg .b8 t;\n\t"
        "  cvt.rn.satfinite.e2m1x2.bf16x2 t, %2;\n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, t;\n\t"
        "  mov.b16 %0, {t, 0}; }"
        : "=h"(out_byte), "=r"(out_dq_bf16x2)
        : "r"(bf16x2_in));
}

static __device__ __forceinline__ void quant_dequant_f32(
    float lo, float hi,
    unsigned short& out_byte,
    unsigned int& out_dq_f16x2)     // dequant f16x2 (no direct f32x2 cvt)
{
    asm("{ .reg .b8 t;\n\t"
        "  cvt.rn.satfinite.e2m1x2.f32 t, %3, %2;\n\t"
        "  cvt.rn.f16x2.e2m1x2 %1, t;\n\t"
        "  mov.b16 %0, {t, 0}; }"
        : "=h"(out_byte), "=r"(out_dq_f16x2)
        : "f"(lo), "f"(hi));
}

#if STOCHASTIC_ROUND
// Stochastic rounding: perturb y by (r - 0.5)*step before cvt.rn.
// `step` varies with magnitude; we approximate using relative perturbation:
//   y_sr = y * (1 + eps) where eps = (r - 0.5) * relative_step
// The e2m1 grid is {0, 0.5, 1, 1.5, 2, 3, 4, 6}. Relative step near any value
// is at most 1/3 (biggest jump is 4→6 = 50% relative, half-step = 25%).
// For simplicity: perturb by (r - 0.5) × 0.0625 of y (empirically OK).
//
// Random source: bottom 7 bits of another bf16 mantissa in the group.
// We pass in one u32 of "random bits" per call and consume 7+7 for the pair.
static __device__ __forceinline__ unsigned int bf16x2_sr_perturb(
    unsigned int bf16x2, unsigned int rand_bits)
{
    // rand_bits: bits[6:0] for lo, bits[13:7] for hi. Take as u7 ∈ [0, 128).
    unsigned int r_lo = rand_bits & 0x7Fu;
    unsigned int r_hi = (rand_bits >> 7) & 0x7Fu;
    // Perturb bf16 mantissa by (r - 64): signed in [-64, 63]
    // Directly adding r to mantissa bits (bf16 has 7 mantissa bits so this maps well)
    // Additionally: bias by -64 (subtract half) for centered SR.
    // Trick: add r_lo into bf16_lo mantissa with signed bias -64:
    //   new_mant = (mantissa + r_lo - 64) & 0x7F  (wraps)
    // For correctness we want bf16 arithmetic: y + (r - 0.5) × ulp. ulp_of_bf16 = 2^(exp-7).
    // Adding r_lo-64 to the mantissa bits (u16 view) adds (r_lo-64) × 2^(exp-7) to the value.
    // That's what we want.
    unsigned int lo = bf16x2 & 0xFFFFu;
    unsigned int hi = bf16x2 >> 16;
    // Signed add of (r_lo - 64) to lo mantissa
    lo = (lo + r_lo - 64u) & 0xFFFFu;
    hi = (hi + r_hi - 64u) & 0xFFFFu;
    return lo | (hi << 16);
}
#endif

// Process one group of 16 bf16/f32 values packed as 8 u32 (bf16x2 each).
// Writes 8 primary FP4 bytes + 8 residual FP4 bytes + 2 scale bytes.
static __device__ __forceinline__ void process_group(
    unsigned int w0, unsigned int w1, unsigned int w2, unsigned int w3,
    unsigned int w4, unsigned int w5, unsigned int w6, unsigned int w7,
    unsigned char out_buf[18])
{
    const unsigned int w_arr[8] = {w0, w1, w2, w3, w4, w5, w6, w7};

    // === 1. Compute per-group absmax in f32 ===
    float absmax = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        absmax = fmaxf(absmax, fabsf(BF16_LO(w_arr[k])));
        absmax = fmaxf(absmax, fabsf(BF16_HI(w_arr[k])));
    }
    absmax = fmaxf(absmax, 1e-30f);  // avoid /0

    // === 2. Primary pass: NC=1 or NC=2, e8m0 scale ===
    // Candidate vals: NC=1 uses {6}, NC=2 uses {6, 4}.
    // For each val, descale_raw = absmax / val × SCALE_OVERRIDE (roughly)
    // Specifically: we want dq_max = 6 × descale = absmax × 6 / val → descale = absmax / val.
    // Actually: s_round = round_to_e8m0(absmax / val × SCALE_OVERRIDE).

    unsigned char e8m0_bytes[PRIMARY_NC];
    float descales[PRIMARY_NC];
    unsigned int primary_packed[PRIMARY_NC];
    unsigned int primary_packed_hi[PRIMARY_NC];
    unsigned int dq_bf16x2_primary[PRIMARY_NC][8];
    float err_acc[PRIMARY_NC] = {0};

    #pragma unroll
    for (int c = 0; c < PRIMARY_NC; ++c) {
        float val = (c == 0) ? 6.0f : 4.0f;
        float s_raw = absmax * rcp_approx_ftz(val) * SCALE_OVERRIDE;
        e8m0_bytes[c] = f32_to_e8m0_byte(s_raw, descales[c]);
        float inv_descale = rcp_approx_ftz(descales[c]);

        // Convert factor to bf16x2 replicated (for HMUL2.bf16x2)
        unsigned int factor_rep;
        asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(factor_rep) : "f"(inv_descale));

        unsigned int packed_lo = 0, packed_hi = 0;

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            unsigned int xpair = w_arr[k];
            // Scale to e2m1 domain
            unsigned int scaled;
            asm("mul.bf16x2 %0, %1, %2;" : "=r"(scaled) : "r"(xpair), "r"(factor_rep));

            #if STOCHASTIC_ROUND
            // Use low mantissa bits of a "neighbour" input as random source:
            // Next-k's bits (wrapping).
            unsigned int rand_src = w_arr[(k + 3) & 7];  // pseudo-distant neighbour
            unsigned int rand_bits = ((rand_src >> 1) ^ (rand_src >> 17)) & 0x3FFFu;
            scaled = bf16x2_sr_perturb(scaled, rand_bits);
            #endif

            unsigned short byte_val;
            unsigned int dq_bf16x2;
            quant_dequant_bf16x2(scaled, byte_val, dq_bf16x2);
            dq_bf16x2_primary[c][k] = dq_bf16x2;

            // Pack into u32 output
            unsigned int b = byte_val & 0xFFu;
            if (k < 4)       packed_lo |= (b << (8 * k));
            else             packed_hi |= (b << (8 * (k - 4)));

            // Accumulate error for winner selection (f32, bf16 dq → f32 via cvt+sub+mag/sq)
            // err contribution = (dq_bf16 - x_bf16)² per element, summed.
            // Compute in bf16 first then convert for cheaper math:
            unsigned int err_bf16x2;
            asm("sub.bf16x2 %0, %1, %2;" : "=r"(err_bf16x2) : "r"(dq_bf16x2), "r"(xpair));

            #if ERROR_METRIC_MAE
            // MAE: accumulate |err| in f32. cvt.f32.bf16 per half.
            unsigned short e_lo = err_bf16x2 & 0xFFFFu;
            unsigned short e_hi = err_bf16x2 >> 16;
            float fe_lo = BF16_LO((unsigned int)e_lo);
            float fe_hi = BF16_LO((unsigned int)e_hi);
            err_acc[c] += fabsf(fe_lo) + fabsf(fe_hi);
            #else
            // L2: err² accumulation via fma.rn.f32.bf16 (f32 out from bf16 mul + f32 addend)
            unsigned short e_lo = err_bf16x2 & 0xFFFFu;
            unsigned short e_hi = err_bf16x2 >> 16;
            float e_sq;
            asm("fma.rn.f32.bf16 %0, %1, %1, %2;" : "=f"(e_sq) : "h"(e_lo), "f"(err_acc[c]));
            err_acc[c] = e_sq;
            asm("fma.rn.f32.bf16 %0, %1, %1, %2;" : "=f"(e_sq) : "h"(e_hi), "f"(err_acc[c]));
            err_acc[c] = e_sq;
            #endif
        }
        primary_packed[c] = packed_lo;
        primary_packed_hi[c] = packed_hi;
    }

    // Pick winner
    int winner = 0;
    #if PRIMARY_NC == 2
    winner = (err_acc[1] < err_acc[0]) ? 1 : 0;
    #endif

    // === 3. Compute residual, residual absmax ===
    // residual = x - dq_primary (in bf16)
    unsigned int resid_bf16x2[8];
    float resid_absmax = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        unsigned int res;
        asm("sub.bf16x2 %0, %1, %2;" : "=r"(res) : "r"(w_arr[k]), "r"(dq_bf16x2_primary[winner][k]));
        resid_bf16x2[k] = res;
        // Extract magnitudes
        float lo = BF16_LO(res);
        float hi = BF16_HI(res);
        resid_absmax = fmaxf(resid_absmax, fmaxf(fabsf(lo), fabsf(hi)));
    }
    resid_absmax = fmaxf(resid_absmax, 1e-30f);

    // === 4. Residual pass: e8m0 scale, quantize residual to e2m1 ===
    // Use val=6 always for NC=1, pick between {6,4} for NC=2.
    unsigned char r_e8m0_bytes[RESIDUAL_NC];
    float r_descales[RESIDUAL_NC];
    unsigned int r_packed_lo[RESIDUAL_NC] = {0};
    unsigned int r_packed_hi[RESIDUAL_NC] = {0};
    float r_err_acc[RESIDUAL_NC] = {0};

    #pragma unroll
    for (int c = 0; c < RESIDUAL_NC; ++c) {
        float val = (c == 0) ? 6.0f : 4.0f;
        float s_raw = resid_absmax * rcp_approx_ftz(val) * SCALE_OVERRIDE;
        r_e8m0_bytes[c] = f32_to_e8m0_byte(s_raw, r_descales[c]);
        float inv_descale = rcp_approx_ftz(r_descales[c]);

        unsigned int factor_rep;
        asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(factor_rep) : "f"(inv_descale));

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            unsigned int scaled;
            asm("mul.bf16x2 %0, %1, %2;" : "=r"(scaled) : "r"(resid_bf16x2[k]), "r"(factor_rep));

            #if STOCHASTIC_ROUND
            unsigned int rand_src = w_arr[(k + 5) & 7];
            unsigned int rand_bits = ((rand_src >> 2) ^ (rand_src >> 18)) & 0x3FFFu;
            scaled = bf16x2_sr_perturb(scaled, rand_bits);
            #endif

            unsigned short byte_val;
            unsigned int dq_bf16x2;
            quant_dequant_bf16x2(scaled, byte_val, dq_bf16x2);

            unsigned int b = byte_val & 0xFFu;
            if (k < 4) r_packed_lo[c] |= (b << (8 * k));
            else       r_packed_hi[c] |= (b << (8 * (k - 4)));

            // Err for residual winner (dq_residual - residual)
            unsigned int err;
            asm("sub.bf16x2 %0, %1, %2;" : "=r"(err) : "r"(dq_bf16x2), "r"(resid_bf16x2[k]));
            unsigned short e_lo = err & 0xFFFFu;
            unsigned short e_hi = err >> 16;

            #if ERROR_METRIC_MAE
            r_err_acc[c] += fabsf(BF16_LO((unsigned int)e_lo)) + fabsf(BF16_LO((unsigned int)e_hi));
            #else
            float e_sq;
            asm("fma.rn.f32.bf16 %0, %1, %1, %2;" : "=f"(e_sq) : "h"(e_lo), "f"(r_err_acc[c]));
            r_err_acc[c] = e_sq;
            asm("fma.rn.f32.bf16 %0, %1, %1, %2;" : "=f"(e_sq) : "h"(e_hi), "f"(r_err_acc[c]));
            r_err_acc[c] = e_sq;
            #endif
        }
    }

    int r_winner = 0;
    #if RESIDUAL_NC == 2
    r_winner = (r_err_acc[1] < r_err_acc[0]) ? 1 : 0;
    #endif

    // === 5. Write outputs: 8 primary bytes + 8 residual bytes + 2 scale bytes ===
    *reinterpret_cast<unsigned int*>(&out_buf[0]) = primary_packed[winner];
    *reinterpret_cast<unsigned int*>(&out_buf[4]) = primary_packed_hi[winner];
    *reinterpret_cast<unsigned int*>(&out_buf[8]) = r_packed_lo[r_winner];
    *reinterpret_cast<unsigned int*>(&out_buf[12]) = r_packed_hi[r_winner];
    out_buf[16] = e8m0_bytes[winner];
    out_buf[17] = r_e8m0_bytes[r_winner];
}

// ---------- main kernel ----------

#ifndef MIN_BLOCKS_PER_SM
#define MIN_BLOCKS_PER_SM 0
#endif

extern "C" __global__ void __launch_bounds__(128, MIN_BLOCKS_PER_SM) kernel(
    const float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    int n_threads, int cols_param, int /*unused*/)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_threads) return;

    // Each thread processes GROUPS_PER_THREAD groups of 16 values.
    // Input per thread: 32B × GPT  (= 16 bf16 × GPT, for INPUT_BF16=1)
    //                   64B × GPT  (= 16 f32  × GPT, for INPUT_BF16=0)
    // Output per thread: 18B × GPT
    if (idx == 0) C[0] = AMAX_CONST;

#if INPUT_BF16
    const unsigned int* pIn = reinterpret_cast<const unsigned int*>(A) + idx * (8 * GROUPS_PER_THREAD);
#else
    const float* pIn_f = A + idx * (16 * GROUPS_PER_THREAD);
#endif
    unsigned char* pOut = reinterpret_cast<unsigned char*>(B) + idx * (18 * GROUPS_PER_THREAD);

#if GROUPS_PER_THREAD == 4
    unsigned int wa0,wa1,wa2,wa3,wa4,wa5,wa6,wa7;
    unsigned int wb0,wb1,wb2,wb3,wb4,wb5,wb6,wb7;
    unsigned int wc0,wc1,wc2,wc3,wc4,wc5,wc6,wc7;
    unsigned int wd0,wd1,wd2,wd3,wd4,wd5,wd6,wd7;

  #if INPUT_BF16
    asm volatile(
        "ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%32];\n\t"
        "ld.global.cg.v8.u32 {%8,%9,%10,%11,%12,%13,%14,%15}, [%33];\n\t"
        "ld.global.cg.v8.u32 {%16,%17,%18,%19,%20,%21,%22,%23}, [%34];\n\t"
        "ld.global.cg.v8.u32 {%24,%25,%26,%27,%28,%29,%30,%31}, [%35];"
        : "=r"(wa0),"=r"(wa1),"=r"(wa2),"=r"(wa3),"=r"(wa4),"=r"(wa5),"=r"(wa6),"=r"(wa7),
          "=r"(wb0),"=r"(wb1),"=r"(wb2),"=r"(wb3),"=r"(wb4),"=r"(wb5),"=r"(wb6),"=r"(wb7),
          "=r"(wc0),"=r"(wc1),"=r"(wc2),"=r"(wc3),"=r"(wc4),"=r"(wc5),"=r"(wc6),"=r"(wc7),
          "=r"(wd0),"=r"(wd1),"=r"(wd2),"=r"(wd3),"=r"(wd4),"=r"(wd5),"=r"(wd6),"=r"(wd7)
        : "l"(pIn), "l"(pIn+8), "l"(pIn+16), "l"(pIn+24));
  #else
    // FP32 input: pack adjacent f32 pairs into bf16x2 via cvt for unified pipeline.
    // Load 16 f32 per group (4 v4.f32), then cvt.rn.bf16x2.f32 (pairs) to get 8 bf16x2 per group.
    #define LD_CVT_GROUP(w0,w1,w2,w3,w4,w5,w6,w7, p) \
        do { \
            float4 r0, r1, r2, r3; \
            asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r0.x),"=f"(r0.y),"=f"(r0.z),"=f"(r0.w) : "l"(p)); \
            asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r1.x),"=f"(r1.y),"=f"(r1.z),"=f"(r1.w) : "l"(p + 4)); \
            asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r2.x),"=f"(r2.y),"=f"(r2.z),"=f"(r2.w) : "l"(p + 8)); \
            asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r3.x),"=f"(r3.y),"=f"(r3.z),"=f"(r3.w) : "l"(p + 12)); \
            asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(w0) : "f"(r0.x), "f"(r0.y)); \
            asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(w1) : "f"(r0.z), "f"(r0.w)); \
            asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(w2) : "f"(r1.x), "f"(r1.y)); \
            asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(w3) : "f"(r1.z), "f"(r1.w)); \
            asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(w4) : "f"(r2.x), "f"(r2.y)); \
            asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(w5) : "f"(r2.z), "f"(r2.w)); \
            asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(w6) : "f"(r3.x), "f"(r3.y)); \
            asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(w7) : "f"(r3.z), "f"(r3.w)); \
        } while (0)

    LD_CVT_GROUP(wa0,wa1,wa2,wa3,wa4,wa5,wa6,wa7, pIn_f + 0);
    LD_CVT_GROUP(wb0,wb1,wb2,wb3,wb4,wb5,wb6,wb7, pIn_f + 16);
    LD_CVT_GROUP(wc0,wc1,wc2,wc3,wc4,wc5,wc6,wc7, pIn_f + 32);
    LD_CVT_GROUP(wd0,wd1,wd2,wd3,wd4,wd5,wd6,wd7, pIn_f + 48);
    #undef LD_CVT_GROUP
  #endif

    unsigned char buf[72];  // 4 × 18 bytes
    process_group(wa0,wa1,wa2,wa3,wa4,wa5,wa6,wa7, &buf[0]);
    process_group(wb0,wb1,wb2,wb3,wb4,wb5,wb6,wb7, &buf[18]);
    process_group(wc0,wc1,wc2,wc3,wc4,wc5,wc6,wc7, &buf[36]);
    process_group(wd0,wd1,wd2,wd3,wd4,wd5,wd6,wd7, &buf[54]);

    // Streaming store 72 bytes (4.5 × v4.u32 = 18 u32s). Use 18 × u32 stores.
    #pragma unroll
    for (int i = 0; i < 18; ++i) {
        unsigned int v = *reinterpret_cast<unsigned int*>(&buf[i * 4]);
        asm("st.global.cs.u32 [%0], %1;" :: "l"(pOut + i*4), "r"(v));
    }
#elif GROUPS_PER_THREAD == 1
    unsigned int wa0,wa1,wa2,wa3,wa4,wa5,wa6,wa7;
  #if INPUT_BF16
    asm volatile("ld.global.cg.v8.u32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
                 : "=r"(wa0),"=r"(wa1),"=r"(wa2),"=r"(wa3),
                   "=r"(wa4),"=r"(wa5),"=r"(wa6),"=r"(wa7) : "l"(pIn));
  #else
    float4 r0, r1, r2, r3;
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r0.x),"=f"(r0.y),"=f"(r0.z),"=f"(r0.w) : "l"(pIn_f));
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r1.x),"=f"(r1.y),"=f"(r1.z),"=f"(r1.w) : "l"(pIn_f + 4));
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r2.x),"=f"(r2.y),"=f"(r2.z),"=f"(r2.w) : "l"(pIn_f + 8));
    asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r3.x),"=f"(r3.y),"=f"(r3.z),"=f"(r3.w) : "l"(pIn_f + 12));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(wa0) : "f"(r0.x), "f"(r0.y));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(wa1) : "f"(r0.z), "f"(r0.w));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(wa2) : "f"(r1.x), "f"(r1.y));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(wa3) : "f"(r1.z), "f"(r1.w));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(wa4) : "f"(r2.x), "f"(r2.y));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(wa5) : "f"(r2.z), "f"(r2.w));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(wa6) : "f"(r3.x), "f"(r3.y));
    asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(wa7) : "f"(r3.z), "f"(r3.w));
  #endif

    unsigned char buf[18];
    process_group(wa0,wa1,wa2,wa3,wa4,wa5,wa6,wa7, buf);

    #pragma unroll
    for (int i = 0; i < 18; ++i) {
        pOut[i] = buf[i];
    }
#else
    #error "Only GROUPS_PER_THREAD=1 or 4 implemented (easy to add more)"
#endif
}

// -------------------- init --------------------
// Seeds input with pseudo-random bf16 values in approximately [-2, 2].
extern "C" __global__ void init(float* A, float* /*B*/, float* /*C*/,
                                int n_threads, int /*cols_param*/, int /*unused*/) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_threads) return;

    constexpr int DPT = 8 * GROUPS_PER_THREAD;  // u32 per thread

#if INPUT_BF16
    unsigned int* a = (unsigned int*)A;
    auto fix = [](unsigned int v) -> unsigned int {
        unsigned int sign = v & 0x8000u;
        unsigned int mant = v & 0x007Fu;
        unsigned int expb = ((v >> 7) & 0xFu) | 0x70u;
        return sign | (expb << 7) | mant;
    };
    #pragma unroll
    for (int k = 0; k < DPT; ++k) {
        unsigned int di = idx * DPT + k;
        unsigned int s  = di * 2654435761u + 1u;
        s = s * 1664525u + 1013904223u;
        a[di] = fix(s & 0xFFFFu) | (fix((s >> 16) & 0xFFFFu) << 16);
    }
#else
    float* a = A;
    #pragma unroll
    for (int k = 0; k < DPT * 2; ++k) {
        unsigned int di = idx * DPT * 2 + k;
        unsigned int s  = di * 2654435761u + 1u;
        s = s * 1664525u + 1013904223u;
        // Normalize to ~[-2, 2] via float conversion of bits
        float v = __int_as_float((s & 0x3FFFFFFFu) | 0x3F800000u) - 1.5f;  // [-0.5, 0.5]
        a[di] = v * 4.0f;
    }
#endif
}
