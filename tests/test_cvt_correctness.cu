// CVT Narrow Format Correctness Test for Blackwell (SM 10.3a)
// Tests all supported CVT narrow format variants with known input values.
// Run with: ./QuickRunCUDA tests/test_cvt_correctness.cu -t 1 -b 1

extern "C" __global__ void kernel(float* A, float* B, float* C, int unused_0, int unused_1, int unused_2) {
    float f1 = 1.0f, f2 = 2.0f, f_nhalf = -0.5f;
    unsigned short n16;
    unsigned int r32;
    unsigned int f16x2_pos = 0x40003C00u;   // f16x2: lo=1.0(0x3C00), hi=2.0(0x4000)
    unsigned int f16x2_neg = 0xB8003C00u;   // f16x2: lo=1.0, hi=-0.5(0xB800)

    printf("=== CVT Narrow Format Correctness (SM 10.3a / B300) ===\n");
    printf("f16 encodings: 1.0=0x3C00, 2.0=0x4000, 0.5=0x3800, -0.5=0xB800\n\n");

    // =========================================================================
    // e2m1x2 (NVFP4: 1s+2e+1m = 4 bits, output is .b8)
    // =========================================================================
    printf("--- e2m1x2 (FP4 / NVFP4) ---\n");

    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t,0}; }"
        : "=h"(n16) : "f"(f1), "f"(f2));
    printf("from f32  (1.0, 2.0)  = 0x%02X\n", n16 & 0xFF);

    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f16x2 t, %1; mov.b16 %0, {t,0}; }"
        : "=h"(n16) : "r"(f16x2_pos));
    printf("from f16x2(1.0, 2.0)  = 0x%02X\n", n16 & 0xFF);

    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.relu.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t,0}; }"
        : "=h"(n16) : "f"(f1), "f"(f_nhalf));
    printf("relu f32  (1.0,-0.5)  = 0x%02X  (hi clamped to 0)\n", n16 & 0xFF);

    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.relu.e2m1x2.f16x2 t, %1; mov.b16 %0, {t,0}; }"
        : "=h"(n16) : "r"(f16x2_neg));
    printf("relu f16x2(1.0,-0.5)  = 0x%02X  (hi clamped to 0)\n", n16 & 0xFF);

    // Round-trip: f32(1.0,2.0) -> e2m1x2 -> f16x2
    unsigned short e2m1_enc;
    asm volatile("{ .reg .b8 t; cvt.rn.satfinite.e2m1x2.f32 t, %2, %1; mov.b16 %0, {t,0}; }"
        : "=h"(e2m1_enc) : "f"(f1), "f"(f2));
    asm volatile("{ .reg .b8 t; mov.b16 {t,_}, %1; cvt.rn.f16x2.e2m1x2 %0, t; }"
        : "=r"(r32) : "h"(e2m1_enc));
    printf("roundtrip (1.0,2.0)   = 0x%08X  (expect 0x40003C00)\n", r32);

    // =========================================================================
    // e4m3x2 (FP8 E4M3: 1s+4e+3m = 8 bits, output is .b16)
    // =========================================================================
    printf("\n--- e4m3x2 (FP8 E4M3) ---\n");

    asm volatile("{cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;}" : "=h"(n16) : "f"(f1), "f"(f2));
    printf("from f32  (1.0, 2.0)  = 0x%04X\n", (unsigned)n16);

    asm volatile("{cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}" : "=h"(n16) : "r"(f16x2_pos));
    printf("from f16x2(1.0, 2.0)  = 0x%04X\n", (unsigned)n16);

    // Round-trip
    unsigned short e4m3_enc;
    asm volatile("{cvt.rn.satfinite.e4m3x2.f16x2 %0, %1;}" : "=h"(e4m3_enc) : "r"(f16x2_pos));
    asm volatile("{cvt.rn.f16x2.e4m3x2 %0, %1;}" : "=r"(r32) : "h"(e4m3_enc));
    printf("roundtrip (1.0,2.0)   = 0x%08X  (expect 0x40003C00)\n", r32);

    // =========================================================================
    // e5m2x2 (FP8 E5M2: 1s+5e+2m = 8 bits, output is .b16)
    // =========================================================================
    printf("\n--- e5m2x2 (FP8 E5M2) ---\n");

    asm volatile("{cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;}" : "=h"(n16) : "f"(f1), "f"(f2));
    printf("from f32  (1.0, 2.0)  = 0x%04X\n", (unsigned)n16);

    asm volatile("{cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;}" : "=h"(n16) : "r"(f16x2_pos));
    printf("from f16x2(1.0, 2.0)  = 0x%04X\n", (unsigned)n16);

    unsigned short e5m2_enc;
    asm volatile("{cvt.rn.satfinite.e5m2x2.f16x2 %0, %1;}" : "=h"(e5m2_enc) : "r"(f16x2_pos));
    asm volatile("{cvt.rn.f16x2.e5m2x2 %0, %1;}" : "=r"(r32) : "h"(e5m2_enc));
    printf("roundtrip (1.0,2.0)   = 0x%08X  (expect 0x40003C00)\n", r32);

    // =========================================================================
    // e2m3x2 (MXFP6: 1s+2e+3m = 6 bits, output is .b16, from f32 ONLY)
    // =========================================================================
    printf("\n--- e2m3x2 (FP6 E2M3, from f32 only) ---\n");

    asm volatile("{cvt.rn.satfinite.e2m3x2.f32 %0, %2, %1;}" : "=h"(n16) : "f"(f1), "f"(f2));
    printf("from f32  (1.0, 2.0)  = 0x%04X\n", (unsigned)n16);

    unsigned short e2m3_enc;
    asm volatile("{cvt.rn.satfinite.e2m3x2.f32 %0, %2, %1;}" : "=h"(e2m3_enc) : "f"(f1), "f"(f2));
    asm volatile("{cvt.rn.f16x2.e2m3x2 %0, %1;}" : "=r"(r32) : "h"(e2m3_enc));
    printf("roundtrip (1.0,2.0)   = 0x%08X  (expect ~0x40003C00)\n", r32);

    // =========================================================================
    // e3m2x2 (MXFP6: 1s+3e+2m = 6 bits, output is .b16, from f32 ONLY)
    // =========================================================================
    printf("\n--- e3m2x2 (FP6 E3M2, from f32 only) ---\n");

    asm volatile("{cvt.rn.satfinite.e3m2x2.f32 %0, %2, %1;}" : "=h"(n16) : "f"(f1), "f"(f2));
    printf("from f32  (1.0, 2.0)  = 0x%04X\n", (unsigned)n16);

    unsigned short e3m2_enc;
    asm volatile("{cvt.rn.satfinite.e3m2x2.f32 %0, %2, %1;}" : "=h"(e3m2_enc) : "f"(f1), "f"(f2));
    asm volatile("{cvt.rn.f16x2.e3m2x2 %0, %1;}" : "=r"(r32) : "h"(e3m2_enc));
    printf("roundtrip (1.0,2.0)   = 0x%08X  (expect ~0x40003C00)\n", r32);

    // =========================================================================
    // ue8m0x2 (scale factors, .b16, to/from bf16x2)
    // =========================================================================
    printf("\n--- ue8m0x2 (scale factors) ---\n");
    unsigned int bf16x2_in = 0x3F803F80u;  // bf16x2: 1.0, 1.0
    asm volatile("{cvt.rz.satfinite.ue8m0x2.bf16x2 %0, %1;}" : "=h"(n16) : "r"(bf16x2_in));
    printf("bf16x2(1.0,1.0)->ue8m0x2 = 0x%04X\n", (unsigned)n16);

    asm volatile("{cvt.rn.bf16x2.ue8m0x2 %0, %1;}" : "=r"(r32) : "h"(n16));
    printf("roundtrip ue8m0x2->bf16x2 = 0x%08X  (expect 0x3F803F80)\n", r32);

    printf("\n=== Instruction availability summary ===\n");
    printf("To-narrow from f16x2:  e2m1x2[+relu] e4m3x2 e5m2x2\n");
    printf("To-narrow from f32:    e2m1x2[+relu] e4m3x2 e5m2x2 e2m3x2 e3m2x2\n");
    printf("From-narrow to f16x2:  e2m1x2 e4m3x2 e5m2x2 e2m3x2 e3m2x2\n");
    printf("Scale factors:         ue8m0x2 <-> bf16x2\n");
    printf("NOT supported on sm_103a: bf16x2 src (except ue8m0x2), bf16x2 dest (except ue8m0x2)\n");
    printf("                          e2m3x2/e3m2x2 from f16x2\n");
}
