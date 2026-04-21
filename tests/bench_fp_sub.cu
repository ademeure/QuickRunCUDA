// Does fp sub emit as FFMA?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    float fb = 1.000001f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // sub.rn.f32
        asm("sub.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(fb));
#elif MODE == 1
        // C-style v - b
        fv = fv - fb;
#elif MODE == 2
        // negate then add (a + (-b))
        asm("add.rn.f32 %0, %0, %1;" : "+f"(fv) : "f"(-fb));
#elif MODE == 3
        // fma.rn.f32 with -1.0 multiplier
        asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(1.0f), "f"(-fb));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
