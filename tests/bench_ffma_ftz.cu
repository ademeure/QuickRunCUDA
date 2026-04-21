// FFMA .ftz (flush-to-zero) modifier
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    float fb = 1.000001f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Default fma.rn.f32 (compiler picks)
        fv = fmaf(fv, fb, fv);
#elif MODE == 1
        // PTX explicit fma.rn.ftz.f32 (FTZ enabled)
        asm("fma.rn.ftz.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 2
        // PTX no-ftz (might emit slow fma.rn.f32 without FTZ)
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 3
        // Saturating .sat (clamp to [0,1])
        asm("fma.rn.ftz.sat.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
