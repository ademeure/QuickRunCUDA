// fp32 -> bf16 conversion throughput
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    unsigned int packed = 0;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Scalar fp32 -> bf16 via cvt.rn
        unsigned short bf;
        asm("cvt.rn.bf16.f32 %0, %1;" : "=h"(bf) : "f"(fv));
        packed = packed * 31u + bf;
#elif MODE == 1
        // Packed cvt.rn.bf16x2.f32 (2 fp32 -> 2 bf16 in 1 inst)
        unsigned int bf2;
        asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(bf2) : "f"(fv), "f"(fv * 2.0f));
        packed = packed * 31u + bf2;
#elif MODE == 2
        // Packed satfinite
        unsigned int bf2;
        asm("cvt.rn.satfinite.bf16x2.f32 %0, %1, %2;" : "=r"(bf2) : "f"(fv), "f"(fv * 2.0f));
        packed = packed * 31u + bf2;
#elif MODE == 3
        // Manual bit truncate (simpler, no rounding)
        unsigned int bf = __float_as_uint(fv) >> 16;
        packed = packed * 31u + bf;
#endif
    }

    if (packed == (unsigned)seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = packed;
}
