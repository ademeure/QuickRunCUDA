// sqrt variants: __sqrtf vs sqrtf vs rsqrt+invert
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 1.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // __sqrtf intrinsic (fast)
        fv = __sqrtf(fv);
#elif MODE == 1
        // sqrtf standard (IEEE)
        fv = sqrtf(fv);
#elif MODE == 2
        // PTX sqrt.approx
        asm("sqrt.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 3
        // rsqrt + reciprocal trick: sqrt(x) = x * rsqrt(x)
        float r;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(fv));
        fv = fv * r;
#elif MODE == 4
        // 1/rsqrt
        float r;
        asm("rsqrt.approx.f32 %0, %1;" : "=f"(r) : "f"(fv));
        asm("rcp.approx.f32 %0, %0;" : "+f"(r));
        fv = r;
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
