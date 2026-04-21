// FABS / FNEG cost (often 0 cy via operand modifier)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    float fb = 1.000001f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Pure FFMA chain (baseline)
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 1
        // FFMA with abs modifier (may be free via SASS modifier)
        asm("fma.rn.f32 %0, |%0|, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 2
        // FFMA with neg modifier
        asm("fma.rn.f32 %0, -%0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 3
        // Explicit fabs() before FFMA
        fv = fabsf(fv);
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 4
        // Explicit -fv before FFMA
        fv = -fv;
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
