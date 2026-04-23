// nvfp4_2d_delayed.cu
//
// 2D BF16/FP32 → NVFP4 (e2m1 payload + ue4m3 per-16-group scale) with
// DELAYED per-tensor scaling.
//
//   * Input matrix is M rows × K cols. K must be divisible by 16.
//   * Each 16-element group along K gets its own e4m3 scale byte.
//   * A per-tensor scale (fp32 scalar) is passed IN (from previous iteration).
//     It's used to normalize the per-group scales (sets global_scale).
//   * The kernel computes the CURRENT tensor absmax as a side output
//     (written to C[1]) for use by the NEXT iteration — not the current one.
//     This is "delayed scaling": current quantization uses prev amax only.
//
// Compile-time flags (-H):
//   INPUT_BF16        1 = BF16 input (default), 0 = FP32 input
//   NUM_CANDIDATES    1 or 2 (default 2)
//   STOCHASTIC_ROUND  0 = RTN (default), 1 = SR (random from neighbour data)
//   ERROR_METRIC_MAE  0 = L2 winner (RMSE-optimal), 1 = L1 winner (MAE)
//   GROUPS_PER_THREAD 1 or 4 (default 4)
//
// Output layout (written to B):
//   B[0 .. M*K/2 - 1]                 FP4 bytes  (row-major, K/2 bytes per row)
//   B[M*K/2 .. M*K/2 + M*K/16 - 1]    e4m3 scales (row-major, K/16 bytes per row)
//
// C layout:
//   C[0]  IN:  previous tensor absmax  (fp32, float bit pattern)
//   C[1]  OUT: new tensor absmax       (written by this kernel, for NEXT iter)
//
// Args:
//   A: input tensor
//   B: output (FP4 bytes + scale bytes, see layout above)
//   C: prev/new tensor absmax
//   n_threads  = total number of groups × blocks (see launch geom)
//   cols_param = K (columns)
//   unused     = unused

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#ifndef INPUT_BF16
#define INPUT_BF16 1
#endif
#ifndef NUM_CANDIDATES
#define NUM_CANDIDATES 2
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

#define VSIZE        16
#define FP4_BYTES     8
#define SCALE_OVERRIDE 1.0f
// Per-tensor "global scale" formula for NVFP4:
//   global_scale = tensor_amax × (1/256) × (SCALE_OVERRIDE/6)
// Using tensor_amax provided via C[0] (delayed / prev-iter).

// ---------- helpers ----------

#define BF16_LO(w) __int_as_float((unsigned int)(w) << 16)
#define BF16_HI(w) __int_as_float((unsigned int)(w) & 0xFFFF0000u)

static __device__ __forceinline__ float rcp_approx_ftz(float a) {
    float b;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(b) : "f"(a));
    return b;
}

// e4m3 roundtrip: f32 → e4m3 byte and back to f32 (via f16 intermediate).
static __device__ __forceinline__ float roundtrip_e4m3(float x, unsigned char& out_byte) {
    unsigned short packed;
    asm("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
        : "=h"(packed) : "f"(x), "f"(0.0f));
    out_byte = (unsigned char)(packed & 0xFFu);
    unsigned int f16_pair;
    asm("{cvt.rn.f16x2.e4m3x2 %0, %1;}" : "=r"(f16_pair) : "h"(packed));
    __half_raw hr; hr.x = (unsigned short)(f16_pair & 0xFFFFu);
    return __half2float(__half(hr));
}

// Fused quant+dequant (bf16x2 → e2m1x2 byte + bf16x2 dequant)
static __device__ __forceinline__ void quant_dequant_bf16x2(
    unsigned int bf16x2_in,
    unsigned short& out_byte,
    unsigned int& out_dq_bf16x2)
{
    asm("{ .reg .b8 t;\n\t"
        "  cvt.rn.satfinite.e2m1x2.bf16x2 t, %2;\n\t"
        "  cvt.rn.bf16x2.e2m1x2 %1, t;\n\t"
        "  mov.b16 %0, {t, 0}; }"
        : "=h"(out_byte), "=r"(out_dq_bf16x2)
        : "r"(bf16x2_in));
}

#if STOCHASTIC_ROUND
// SR via signed mantissa perturbation of bf16x2 (same as kernel 1).
static __device__ __forceinline__ unsigned int bf16x2_sr_perturb(
    unsigned int bf16x2, unsigned int rand_bits)
{
    unsigned int r_lo = rand_bits & 0x7Fu;
    unsigned int r_hi = (rand_bits >> 7) & 0x7Fu;
    unsigned int lo = bf16x2 & 0xFFFFu;
    unsigned int hi = bf16x2 >> 16;
    lo = (lo + r_lo - 64u) & 0xFFFFu;
    hi = (hi + r_hi - 64u) & 0xFFFFu;
    return lo | (hi << 16);
}
#endif

// atomicMax on float via u32 (works for non-negative floats: u32 sort same as f32 sort)
static __device__ __forceinline__ float atomic_max_f32_pos(float* addr, float val) {
    unsigned int* addr_u = reinterpret_cast<unsigned int*>(addr);
    unsigned int v_u = __float_as_uint(fmaxf(val, 0.f));
    unsigned int prev = atomicMax(addr_u, v_u);
    return __uint_as_float(prev);
}

// Process a single 16-element group:
//   Inputs: w0..w7 (bf16x2 packed), global_scale (f32, from prev tensor amax),
//           out_byte_addr (where 8 FP4 bytes go), scale_addr (where e4m3 byte goes)
// Returns per-group absmax (f32) for tensor-amax reduction.
static __device__ __forceinline__ float process_group_2d(
    unsigned int w0, unsigned int w1, unsigned int w2, unsigned int w3,
    unsigned int w4, unsigned int w5, unsigned int w6, unsigned int w7,
    float global_scale,
    unsigned char* __restrict__ fp4_out,
    unsigned char* __restrict__ scale_out,
    unsigned int rand_src_u32)
{
    const unsigned int w_arr[8] = {w0, w1, w2, w3, w4, w5, w6, w7};

    // Absmax
    float absmax = 0.f;
    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        absmax = fmaxf(absmax, fabsf(BF16_LO(w_arr[k])));
        absmax = fmaxf(absmax, fabsf(BF16_HI(w_arr[k])));
    }
    float group_absmax_return = absmax;
    absmax = fmaxf(absmax, 1e-30f);

    float inv_scale = rcp_approx_ftz(global_scale);

    // NC candidate processing
    unsigned char fp8_bytes[NUM_CANDIDATES];
    unsigned int packed_lo[NUM_CANDIDATES] = {0};
    unsigned int packed_hi[NUM_CANDIDATES] = {0};
    float err_acc[NUM_CANDIDATES] = {0};

    #pragma unroll
    for (int c = 0; c < NUM_CANDIDATES; ++c) {
        float val = (c == 0) ? 6.0f : 4.0f;
        float s_raw = absmax * rcp_approx_ftz(val) * SCALE_OVERRIDE * inv_scale;
        float s_round = roundtrip_e4m3(s_raw, fp8_bytes[c]);
        float descale = s_round * global_scale + 1e-30f;
        float factor = rcp_approx_ftz(descale);

        unsigned int factor_rep;
        asm("cvt.rn.bf16x2.f32 %0, %1, %1;" : "=r"(factor_rep) : "f"(factor));

        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            unsigned int xpair = w_arr[k];
            unsigned int scaled;
            asm("mul.bf16x2 %0, %1, %2;" : "=r"(scaled) : "r"(xpair), "r"(factor_rep));

            #if STOCHASTIC_ROUND
            unsigned int rand_bits = ((rand_src_u32 >> k) ^ (rand_src_u32 >> (k + 13))) & 0x3FFFu;
            scaled = bf16x2_sr_perturb(scaled, rand_bits);
            #endif

            unsigned short byte_val;
            unsigned int dq_bf16x2;
            quant_dequant_bf16x2(scaled, byte_val, dq_bf16x2);

            unsigned int b = byte_val & 0xFFu;
            if (k < 4) packed_lo[c] |= (b << (8 * k));
            else       packed_hi[c] |= (b << (8 * (k - 4)));

            // Error accumulation
            unsigned int err_bf16x2;
            asm("sub.bf16x2 %0, %1, %2;" : "=r"(err_bf16x2) : "r"(dq_bf16x2), "r"(xpair));
            unsigned short e_lo = err_bf16x2 & 0xFFFFu;
            unsigned short e_hi = err_bf16x2 >> 16;

            #if ERROR_METRIC_MAE
            err_acc[c] += fabsf(BF16_LO((unsigned int)e_lo)) + fabsf(BF16_LO((unsigned int)e_hi));
            #else
            float tmp = err_acc[c];
            asm("fma.rn.f32.bf16 %0, %1, %1, %2;" : "=f"(tmp) : "h"(e_lo), "f"(tmp));
            asm("fma.rn.f32.bf16 %0, %1, %1, %2;" : "=f"(tmp) : "h"(e_hi), "f"(tmp));
            err_acc[c] = tmp;
            #endif
        }
    }

    // Winner
    int winner = 0;
    #if NUM_CANDIDATES == 2
    winner = (err_acc[1] < err_acc[0]) ? 1 : 0;
    #endif

    // Write outputs
    *reinterpret_cast<unsigned int*>(&fp4_out[0]) = packed_lo[winner];
    *reinterpret_cast<unsigned int*>(&fp4_out[4]) = packed_hi[winner];
    scale_out[0] = fp8_bytes[winner];

    return group_absmax_return;
}

// ---------- kernel ----------

#ifndef MIN_BLOCKS_PER_SM
#define MIN_BLOCKS_PER_SM 0
#endif

extern "C" __global__ void __launch_bounds__(128, MIN_BLOCKS_PER_SM) kernel(
    const float* __restrict__ A,
    float* __restrict__ B,
    float* __restrict__ C,
    int n_threads, int K, int /*unused*/)
{
    const int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= n_threads) return;

    // Load previous tensor absmax (delayed scaling): C[0].
    // Compute global_scale once per block (broadcast via shared or just reload).
    // For simplicity every thread computes it from the same constant memory.
    float tensor_amax_prev = C[0];
    // Default: if C[0] is 0 or negative (uninitialised), use AMAX_CONST=1.3.
    if (!(tensor_amax_prev > 0.f)) tensor_amax_prev = 1.3f;
    const float global_scale = tensor_amax_prev * (1.f / 256.f) * (SCALE_OVERRIDE / 6.f);

    // Each thread processes GROUPS_PER_THREAD consecutive groups along K.
    // Thread idx handles groups [idx*GPT .. idx*GPT + GPT).
    // Group g in flat order maps to row = g / (K/16), col = (g % (K/16)) × 16.

    const int GPT = GROUPS_PER_THREAD;
    const int group_start = idx * GPT;

    // Input pointers: BF16 means 16 bf16 per group = 8 u32 per group = 8×GPT u32 per thread
#if INPUT_BF16
    const unsigned int* pIn = reinterpret_cast<const unsigned int*>(A) + group_start * 8;
#else
    const float* pIn_f = A + group_start * 16;
#endif

    // Output pointers
    // FP4 bytes: 8 bytes per group, stored contiguously
    unsigned char* p_fp4 = reinterpret_cast<unsigned char*>(B) + group_start * 8;
    // Scale bytes: 1 byte per group, stored after all FP4 data
    // Total groups = n_threads * GPT (assumes all threads valid)
    const int total_groups = n_threads * GPT;
    unsigned char* p_scale = reinterpret_cast<unsigned char*>(B) + total_groups * 8 + group_start;

    // Random source for SR: low bits of group_start input dword
    unsigned int rand_src = group_start * 2654435761u + (unsigned int)idx;

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
    // FP32 → convert to bf16x2 pairs upfront for unified pipeline
    auto load_and_cvt = [](const float* p, unsigned int& o0, unsigned int& o1,
                           unsigned int& o2, unsigned int& o3,
                           unsigned int& o4, unsigned int& o5,
                           unsigned int& o6, unsigned int& o7) {
        float4 r0, r1, r2, r3;
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r0.x),"=f"(r0.y),"=f"(r0.z),"=f"(r0.w) : "l"(p));
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r1.x),"=f"(r1.y),"=f"(r1.z),"=f"(r1.w) : "l"(p + 4));
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r2.x),"=f"(r2.y),"=f"(r2.z),"=f"(r2.w) : "l"(p + 8));
        asm volatile("ld.global.cg.v4.f32 {%0,%1,%2,%3}, [%4];" : "=f"(r3.x),"=f"(r3.y),"=f"(r3.z),"=f"(r3.w) : "l"(p + 12));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o0) : "f"(r0.x), "f"(r0.y));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o1) : "f"(r0.z), "f"(r0.w));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o2) : "f"(r1.x), "f"(r1.y));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o3) : "f"(r1.z), "f"(r1.w));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o4) : "f"(r2.x), "f"(r2.y));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o5) : "f"(r2.z), "f"(r2.w));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o6) : "f"(r3.x), "f"(r3.y));
        asm("cvt.rn.bf16x2.f32 %0, %2, %1;" : "=r"(o7) : "f"(r3.z), "f"(r3.w));
    };
    load_and_cvt(pIn_f +  0, wa0,wa1,wa2,wa3,wa4,wa5,wa6,wa7);
    load_and_cvt(pIn_f + 16, wb0,wb1,wb2,wb3,wb4,wb5,wb6,wb7);
    load_and_cvt(pIn_f + 32, wc0,wc1,wc2,wc3,wc4,wc5,wc6,wc7);
    load_and_cvt(pIn_f + 48, wd0,wd1,wd2,wd3,wd4,wd5,wd6,wd7);
  #endif

    float amax0 = process_group_2d(wa0,wa1,wa2,wa3,wa4,wa5,wa6,wa7,
                                    global_scale, p_fp4 + 0,  p_scale + 0, rand_src);
    float amax1 = process_group_2d(wb0,wb1,wb2,wb3,wb4,wb5,wb6,wb7,
                                    global_scale, p_fp4 + 8,  p_scale + 1, rand_src ^ 0x55555555u);
    float amax2 = process_group_2d(wc0,wc1,wc2,wc3,wc4,wc5,wc6,wc7,
                                    global_scale, p_fp4 + 16, p_scale + 2, rand_src ^ 0xAAAAAAAAu);
    float amax3 = process_group_2d(wd0,wd1,wd2,wd3,wd4,wd5,wd6,wd7,
                                    global_scale, p_fp4 + 24, p_scale + 3, rand_src ^ 0xCCCCCCCCu);

    // Reduce thread-local max across groups
    float thread_amax = fmaxf(fmaxf(amax0, amax1), fmaxf(amax2, amax3));

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
    float thread_amax = process_group_2d(wa0,wa1,wa2,wa3,wa4,wa5,wa6,wa7,
                                         global_scale, p_fp4, p_scale, rand_src);
#else
    #error "Only GROUPS_PER_THREAD=1 or 4 implemented"
#endif

    // Warp-level reduce max (32 lanes)
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        thread_amax = fmaxf(thread_amax, __shfl_xor_sync(0xFFFFFFFFu, thread_amax, offset));
    }

    // Block-level reduce via shared: each warp's lane-0 writes, warp 0 reduces, one atomic per block.
    __shared__ float s_warp_amax[4];  // blockDim.x=128 → 4 warps
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) s_warp_amax[warp] = thread_amax;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 4) ? s_warp_amax[lane] : 0.f;
        #pragma unroll
        for (int offset = 2; offset > 0; offset >>= 1) {
            v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, offset));
        }
        if (lane == 0) atomic_max_f32_pos(&C[1], v);
    }
}

// -------------------- init --------------------
extern "C" __global__ void init(float* A, float* /*B*/, float* C,
                                int n_threads, int /*cols_param*/, int /*unused*/) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx == 0) {
        C[0] = 1.5f;      // previous tensor amax (default prior)
        C[1] = 0.0f;      // start-of-iteration reset for new amax
    }
    if (idx >= n_threads) return;

    constexpr int DPT = 8 * GROUPS_PER_THREAD;
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
        float v = __int_as_float((s & 0x3FFFFFFFu) | 0x3F800000u) - 1.5f;
        a[di] = v * 4.0f;
    }
#endif
}
