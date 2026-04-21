// FMUL vs FFMA (does removing addend help?)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 1.0f + (float)u2 * 1e-9f;
    float fb = 1.000001f;
    float fc = 0.5f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // FMUL only - 2 sources
        asm("mul.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(fb));
#elif MODE == 1
        // FADD only - 2 sources
        asm("add.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(fb));
#elif MODE == 2
        // FFMA 2-source: a*b+a (chain in a, second source = b)
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 3
        // FFMA 3-source distinct
        asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(fc));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
