// Emit every CVT variant of interest. Accumulate every output into acc to keep
// all instructions live. SASS will have them in order.
#define OH(pref, ...) do { \
    unsigned short _oh; \
    asm volatile(pref " %0, " __VA_ARGS__ : "=h"(_oh) : "f"(f32), "f"(f32b), "r"(f16x2), "h"(narrow16), "r"(rbits)); \
    acc ^= (unsigned int)_oh; } while(0)

extern "C" __global__ void kernel(float* A, float* B, float* C,
                                  int seed, int u1, int u2) {
    float    f32 = (float)threadIdx.x + 1.0f;
    float    f32b = f32 + 0.5f;
    unsigned int   f16x2 = 0x3C003C01u ^ threadIdx.x;
    unsigned int   bf16x2 = 0x3F803F81u ^ threadIdx.x;
    unsigned short f16    = 0x3C00u ^ (unsigned)threadIdx.x;
    unsigned short bf16   = 0x3F80u ^ (unsigned)threadIdx.x;
    unsigned short narrow16 = 0x4242u ^ (unsigned)threadIdx.x;
    unsigned int   rbits  = 0xDEADBEEFu ^ threadIdx.x;

    unsigned int acc = 0;
    unsigned short oh; unsigned int or32; float wf;

    #define EMIT_H(code) asm volatile(code : "=h"(oh) : "f"(f32), "f"(f32b), "r"(f16x2), "r"(bf16x2), "h"(f16), "h"(bf16), "h"(narrow16), "r"(rbits)); acc ^= (unsigned int)oh
    #define EMIT_R(code) asm volatile(code : "=r"(or32) : "f"(f32), "f"(f32b), "r"(f16x2), "r"(bf16x2), "h"(f16), "h"(bf16), "h"(narrow16), "r"(rbits)); acc ^= or32

    // ======= SCALAR =======
    EMIT_H("cvt.rn.satfinite.f16.f32 %0, %1;");                 // 1.  f32 → f16 (satfinite)
    EMIT_H("cvt.rn.satfinite.bf16.f32 %0, %1;");                // 2.  f32 → bf16 (satfinite)
    EMIT_H("cvt.rn.f16.f32 %0, %1;");                           // 3.  f32 → f16 (non-satfinite, SLOW)
    EMIT_H("cvt.rn.bf16.f32 %0, %1;");                          // 4.  f32 → bf16 (non-satfinite)
    EMIT_R("cvt.rn.satfinite.tf32.f32 %0, %1;");                // 5.  f32 → tf32 (rn)
    EMIT_R("cvt.rna.satfinite.tf32.f32 %0, %1;");               // 6.  f32 → tf32 (rna — emulated)

    // ======= x2 — scalar f32 pair source =======
    EMIT_R("cvt.rn.f16x2.f32 %0, %2, %1;");                     // 7.  f32,f32 → f16x2
    EMIT_R("cvt.rn.bf16x2.f32 %0, %2, %1;");                    // 8.  f32,f32 → bf16x2
    EMIT_H("cvt.rn.satfinite.e4m3x2.f32 %0, %2, %1;");          // 9.  f32,f32 → e4m3x2
    EMIT_H("cvt.rn.satfinite.e5m2x2.f32 %0, %2, %1;");          // 10. f32,f32 → e5m2x2
    EMIT_H("cvt.rn.satfinite.e2m3x2.f32 %0, %2, %1;");          // 11. f32,f32 → e2m3x2 (FP6)
    EMIT_H("cvt.rn.satfinite.e3m2x2.f32 %0, %2, %1;");          // 12. f32,f32 → e3m2x2 (FP6)

    // ======= x2 — packed from f16x2 =======
    EMIT_H("cvt.rn.satfinite.e4m3x2.f16x2 %0, %3;");            // 13. f16x2 → e4m3x2
    EMIT_H("cvt.rn.satfinite.e5m2x2.f16x2 %0, %3;");            // 14. f16x2 → e5m2x2

    // ======= x2 — unpack to f16x2 =======
    EMIT_R("cvt.rn.f16x2.e4m3x2 %0, %7;");                      // 15. e4m3x2 → f16x2
    EMIT_R("cvt.rn.f16x2.e5m2x2 %0, %7;");                      // 16. e5m2x2 → f16x2
    EMIT_R("cvt.rn.f16x2.e2m3x2 %0, %7;");                      // 17. e2m3x2 → f16x2 (FP6)
    EMIT_R("cvt.rn.f16x2.e3m2x2 %0, %7;");                      // 18. e3m2x2 → f16x2 (FP6)

    // ======= x2 — FP4 (e2m1x2) with .b8 pack/unpack mov =======
    // pack: .b8 result needs mov.b16 wrap
    asm volatile("{ .reg .b8 _b; cvt.rn.satfinite.e2m1x2.f32 _b, %2, %1; mov.b16 %0,{_b,0}; }"
                 : "=h"(oh) : "f"(f32), "f"(f32b));
    acc ^= (unsigned int)oh;
    asm volatile("{ .reg .b8 _b; cvt.rn.satfinite.e2m1x2.f16x2 _b, %1; mov.b16 %0,{_b,0}; }"
                 : "=h"(oh) : "r"(f16x2));
    acc ^= (unsigned int)oh;
    // unpack
    asm volatile("{ .reg .b8 _b; mov.b16 {_b,_}, %1; cvt.rn.f16x2.e2m1x2 %0, _b; }"
                 : "=r"(or32) : "h"(narrow16));
    acc ^= or32;

    // ======= x4 stochastic rounding =======
    asm volatile("cvt.rs.satfinite.e4m3x4.f32 %0, {%2,%3,%4,%5}, %1;"
                 : "=r"(or32)
                 : "r"(rbits), "f"(f32), "f"(f32b), "f"(f32+0.25f), "f"(f32+0.75f));
    acc ^= or32;
    asm volatile("cvt.rs.satfinite.e5m2x4.f32 %0, {%2,%3,%4,%5}, %1;"
                 : "=r"(or32)
                 : "r"(rbits), "f"(f32), "f"(f32b), "f"(f32+0.25f), "f"(f32+0.75f));
    acc ^= or32;
    // e2m1x4 (FP4 SR) has .b16 output
    asm volatile("{ .reg .b16 _h; cvt.rs.satfinite.e2m1x4.f32 _h, {%2,%3,%4,%5}, %1; mov.b16 %0, _h; }"
                 : "=h"(oh)
                 : "r"(rbits), "f"(f32), "f"(f32b), "f"(f32+0.25f), "f"(f32+0.75f));
    acc ^= (unsigned int)oh;

    // ======= Widen f16/bf16 → f32 =======
    asm volatile("cvt.f32.f16 %0, %1;" : "=f"(wf) : "h"(f16));
    acc ^= __float_as_int(wf);
    asm volatile("cvt.rn.f32.bf16 %0, %1;" : "=f"(wf) : "h"(bf16));
    acc ^= __float_as_int(wf);

    // ======= UE8M0 =======
    EMIT_H("cvt.rz.satfinite.ue8m0x2.bf16x2 %0, %4;");          // 26. bf16x2 → ue8m0x2
    EMIT_H("cvt.rz.satfinite.ue8m0x2.f32 %0, %2, %1;");         // 27. f32,f32 → ue8m0x2
    EMIT_R("cvt.rn.bf16x2.ue8m0x2 %0, %7;");                    // 28. ue8m0x2 → bf16x2

    if ((int)acc == seed) ((unsigned int*)C)[threadIdx.x] = acc;
}
