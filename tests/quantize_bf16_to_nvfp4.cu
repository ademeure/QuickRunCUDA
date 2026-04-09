// BF16 -> NVFP4 (e2m1) Quantization Kernel
// Each thread: 256-bit load (16 BF16) -> absmax -> scale -> CVT e2m1 -> store
//
// Compile-time options via -H:
//   SCALE_E4M3  - output e4m3 scale factor (default)
//   SCALE_BF16  - output bf16 scale factor
//
// Input layout:  A[] = BF16 values, contiguous, length N (padded to 16)
// Output layout: C[] = packed {fp4_data[8], scale[1 or 2]} per group of 16 elements
//
// Usage: ./QuickRunCUDA tests/quantize_bf16_to_nvfp4.cu -r -A <N> -C <N> -t 256 -b <N/16/256>

// e2m1 max representable value
#define E2M1_MAX 6.0f
#define E2M1_INV_MAX (1.0f / 6.0f)

extern "C" __global__ void kernel(const float* __restrict__ A,
                                  float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N, int unused_1, int unused_2) {
    // A is actually BF16 data reinterpreted as float* by QuickRunCUDA
    // C is output buffer for packed FP4 + scale

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // N is not used -- grid is sized exactly to cover the data
    // Each thread handles 16 BF16 values (256 bits = 2x int4 loads)

    // =========================================================================
    // 1. LOAD: 256-bit (2x 128-bit) = 16 BF16 values
    // =========================================================================
    const int4* in_ptr = (const int4*)A;
    int4 load0 = in_ptr[tid * 2];      // first 8 BF16 (128 bits)
    int4 load1 = in_ptr[tid * 2 + 1];  // next 8 BF16 (128 bits)

    // 8 packed BF16 pairs (each uint32 holds 2 BF16)
    unsigned int bp0 = (unsigned int)load0.x, bp1 = (unsigned int)load0.y;
    unsigned int bp2 = (unsigned int)load0.z, bp3 = (unsigned int)load0.w;
    unsigned int bp4 = (unsigned int)load1.x, bp5 = (unsigned int)load1.y;
    unsigned int bp6 = (unsigned int)load1.z, bp7 = (unsigned int)load1.w;

    // =========================================================================
    // 2. BF16 -> F32 (free: just shift left 16, compiles to PRMT)
    //    + find absmax across all 16 values
    // =========================================================================
    float f0, f1, f2, f3, f4, f5, f6, f7;
    float f8, f9, f10, f11, f12, f13, f14, f15;

    // Unpack BF16 pairs to F32
    f0  = __int_as_float((bp0 & 0xFFFFu) << 16);
    f1  = __int_as_float(bp0 & 0xFFFF0000u);
    f2  = __int_as_float((bp1 & 0xFFFFu) << 16);
    f3  = __int_as_float(bp1 & 0xFFFF0000u);
    f4  = __int_as_float((bp2 & 0xFFFFu) << 16);
    f5  = __int_as_float(bp2 & 0xFFFF0000u);
    f6  = __int_as_float((bp3 & 0xFFFFu) << 16);
    f7  = __int_as_float(bp3 & 0xFFFF0000u);
    f8  = __int_as_float((bp4 & 0xFFFFu) << 16);
    f9  = __int_as_float(bp4 & 0xFFFF0000u);
    f10 = __int_as_float((bp5 & 0xFFFFu) << 16);
    f11 = __int_as_float(bp5 & 0xFFFF0000u);
    f12 = __int_as_float((bp6 & 0xFFFFu) << 16);
    f13 = __int_as_float(bp6 & 0xFFFF0000u);
    f14 = __int_as_float((bp7 & 0xFFFFu) << 16);
    f15 = __int_as_float(bp7 & 0xFFFF0000u);

    // Absmax via tree reduction (8 fmaxf of abs pairs -> 4 -> 2 -> 1)
    float a0 = fmaxf(fabsf(f0),  fabsf(f1));
    float a1 = fmaxf(fabsf(f2),  fabsf(f3));
    float a2 = fmaxf(fabsf(f4),  fabsf(f5));
    float a3 = fmaxf(fabsf(f6),  fabsf(f7));
    float a4 = fmaxf(fabsf(f8),  fabsf(f9));
    float a5 = fmaxf(fabsf(f10), fabsf(f11));
    float a6 = fmaxf(fabsf(f12), fabsf(f13));
    float a7 = fmaxf(fabsf(f14), fabsf(f15));
    float b0 = fmaxf(a0, a1), b1 = fmaxf(a2, a3);
    float b2 = fmaxf(a4, a5), b3 = fmaxf(a6, a7);
    float c0 = fmaxf(b0, b1), c1 = fmaxf(b2, b3);
    float absmax = fmaxf(c0, c1);

    // =========================================================================
    // 3. COMPUTE SCALE: scale = absmax / 6.0, inv_scale = 6.0 / absmax
    // =========================================================================
    // Use rcp.approx for fast division (matches -use_fast_math behavior)
    float inv_absmax;
    asm volatile("rcp.approx.f32 %0, %1;" : "=f"(inv_absmax) : "f"(absmax));
    float inv_scale = E2M1_MAX * inv_absmax;  // 6.0 / absmax
    float scale = absmax * E2M1_INV_MAX;       // absmax / 6.0

    // Handle zero absmax
    if (absmax == 0.0f) { inv_scale = 0.0f; scale = 0.0f; }

    // =========================================================================
    // 4. PRE-SCALE + CVT to e2m1x2 (8 conversions = 16 FP4 values)
    //    Path: F32 * inv_scale -> cvt.rn.satfinite.e2m1x2.f32
    //    F2FP runs on SFU pipe at 32 ops/SM/clk (full-rate)
    // =========================================================================
    unsigned short q0, q1, q2, q3, q4, q5, q6, q7;

    // Scale values
    f0  *= inv_scale; f1  *= inv_scale;
    f2  *= inv_scale; f3  *= inv_scale;
    f4  *= inv_scale; f5  *= inv_scale;
    f6  *= inv_scale; f7  *= inv_scale;
    f8  *= inv_scale; f9  *= inv_scale;
    f10 *= inv_scale; f11 *= inv_scale;
    f12 *= inv_scale; f13 *= inv_scale;
    f14 *= inv_scale; f15 *= inv_scale;

    // Convert pairs to e2m1x2 (8 F2FP instructions)
    asm volatile("{ .reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, 0}; }"
        : "=h"(q0) : "f"(f0), "f"(f1));
    asm volatile("{ .reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, 0}; }"
        : "=h"(q1) : "f"(f2), "f"(f3));
    asm volatile("{ .reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, 0}; }"
        : "=h"(q2) : "f"(f4), "f"(f5));
    asm volatile("{ .reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, 0}; }"
        : "=h"(q3) : "f"(f6), "f"(f7));
    asm volatile("{ .reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, 0}; }"
        : "=h"(q4) : "f"(f8), "f"(f9));
    asm volatile("{ .reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, 0}; }"
        : "=h"(q5) : "f"(f10), "f"(f11));
    asm volatile("{ .reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, 0}; }"
        : "=h"(q6) : "f"(f12), "f"(f13));
    asm volatile("{ .reg .b8 t;\n\t"
        "cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t, 0}; }"
        : "=h"(q7) : "f"(f14), "f"(f15));

    // =========================================================================
    // 5. PACK + STORE: 8 bytes of FP4 data + scale factor
    // =========================================================================
    // Pack 8 e2m1x2 bytes into a uint2 (64-bit store)
    unsigned int pack_lo = ((unsigned int)(q0 & 0xFF))
                         | ((unsigned int)(q1 & 0xFF) << 8)
                         | ((unsigned int)(q2 & 0xFF) << 16)
                         | ((unsigned int)(q3 & 0xFF) << 24);
    unsigned int pack_hi = ((unsigned int)(q4 & 0xFF))
                         | ((unsigned int)(q5 & 0xFF) << 8)
                         | ((unsigned int)(q6 & 0xFF) << 16)
                         | ((unsigned int)(q7 & 0xFF) << 24);

    // Output layout per thread: [8 bytes FP4 data]
    unsigned int* out = (unsigned int*)C;
    out[tid * 2]     = pack_lo;
    out[tid * 2 + 1] = pack_hi;

    // Scale factor stored separately in B array
#ifdef SCALE_BF16
    // BF16 scale factor (16-bit)
    unsigned short scale_bf16;
    asm volatile("{cvt.rn.bf16.f32 %0, %1;}" : "=h"(scale_bf16) : "f"(scale));
    ((unsigned short*)B)[tid] = scale_bf16;
#else
    // E4M3 scale factor (8-bit) -- default
    unsigned short scale_e4m3;
    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}"
        : "=h"(scale_e4m3) : "f"(scale), "f"(0.0f));
    ((unsigned char*)B)[tid] = (unsigned char)(scale_e4m3 & 0xFF);
#endif
}

// Init kernel: fill A with random BF16 data
extern "C" __global__ void init(float* A, float* B, float* C,
                                int N, int unused_1, int unused_2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    // Fill A with pseudo-random BF16 pairs
    unsigned int seed = tid * 2654435761u + 1;
    unsigned int* a_u32 = (unsigned int*)A;
    int num_pairs = N / 2;  // N is in dwords, each holds 2 BF16
    if (tid < num_pairs) {
        seed = seed * 1664525u + 1013904223u;
        // Generate BF16 pair: values in [-1, 1] range
        // BF16 1.0 = 0x3F80, BF16 -1.0 = 0xBF80
        unsigned int hi = (seed >> 16) & 0xFFFF;
        unsigned int lo = seed & 0xFFFF;
        // Clamp to reasonable BF16 range
        a_u32[tid] = (hi << 16) | lo;
    }
}
