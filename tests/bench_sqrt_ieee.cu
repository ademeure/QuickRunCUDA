// PTX sqrt rounding modes
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 1.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // IEEE-rounded sqrt
        asm("sqrt.rn.f32 %0, %0;" : "+f"(fv));
#elif MODE == 1
        // approx sqrt (MUFU.SQRT)
        asm("sqrt.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 2
        // .ftz approx (flush denormals)
        asm("sqrt.approx.ftz.f32 %0, %0;" : "+f"(fv));
#elif MODE == 3
        // Default sqrtf intrinsic for comparison
        fv = sqrtf(fv);
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
