// FFMA with special FP values (denormals, NaN, inf)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv, fb;

#if MODE == 0
    // Normal values
    fv = (float)threadIdx.x + 1.5f + (float)u2 * 1e-9f;
    fb = 1.000001f;
#elif MODE == 1
    // Denormal values (smaller than 2^-126)
    fv = 1e-40f + (float)u2 * 1e-50f;  // denormal
    fb = 1.0f;
#elif MODE == 2
    // NaN
    fv = __int_as_float(0x7FC00000);  // qNaN
    fb = 1.0f;
#elif MODE == 3
    // Infinity
    fv = __int_as_float(0x7F800000);  // +inf
    fb = 1.0f;
#elif MODE == 4
    // Mixed normal x denormal
    fv = (float)threadIdx.x + 1.5f;
    fb = 1e-40f;  // denormal multiplier — likely flushes
#endif

    for (int i = 0; i < ITERS; i++) {
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
