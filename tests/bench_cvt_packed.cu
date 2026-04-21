// Various packed cvt (per F2FP) - sanity check
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv1 = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    float fv2 = (float)threadIdx.x * 2.0f + (float)u2 * 1e-9f;
    unsigned int packed = 0;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // cvt.rn.f16x2.f32 (2 fp32 -> 2 f16 packed)
        unsigned int p;
        asm("cvt.rn.f16x2.f32 %0, %1, %2;" : "=r"(p) : "f"(fv1), "f"(fv2));
        packed = packed * 31u + p;
#elif MODE == 1
        // cvt.rn.bf16x2.f32 (2 fp32 -> 2 bf16 packed)
        unsigned int p;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(p) : "f"(fv1), "f"(fv2));
        packed = packed * 31u + p;
#elif MODE == 2
        // cvt.rn.satfinite.bf16x2.f32 (with saturation)
        unsigned int p;
        asm("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(p) : "f"(fv1), "f"(fv2));
        packed = packed * 31u + p;
#elif MODE == 3
        // cvt.rn.f16.f32 (single scalar fp32 -> f16)
        unsigned short p;
        asm("cvt.rn.f16.f32 %0, %1;" : "=h"(p) : "f"(fv1));
        packed = packed * 31u + p;
#endif
    }

    if (packed == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = packed;
}
