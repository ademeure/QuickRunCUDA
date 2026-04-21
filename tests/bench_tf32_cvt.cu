// TF32 conversion (truncates fp32 mantissa to 10 bits)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // PTX cvt.rna.tf32.f32 (round to nearest, away from zero)
        unsigned int t;
        asm("cvt.rna.tf32.f32 %0, %1;" : "=r"(t) : "f"(fv));
        fv = __uint_as_float(t);
#elif MODE == 1
        // PTX cvt.rn.tf32.f32
        unsigned int t;
        asm("cvt.rn.tf32.f32 %0, %1;" : "=r"(t) : "f"(fv));
        fv = __uint_as_float(t);
#elif MODE == 2
        // Manual mantissa truncation (zero out low 13 bits)
        unsigned int x = __float_as_uint(fv);
        x = x & 0xFFFFE000u;  // mask low 13 mantissa bits
        fv = __uint_as_float(x);
#elif MODE == 3
        // Baseline: just chain
        fv = fv * 1.000001f;
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
