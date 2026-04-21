// Does .ftz unlock a fast path for MUFU ops beyond sqrt?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 1.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // rsqrt.approx (NO .ftz)
        asm("rsqrt.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 1
        // rsqrt.approx.ftz
        asm("rsqrt.approx.ftz.f32 %0, %0;" : "+f"(fv));
#elif MODE == 2
        // rcp.approx (NO .ftz)
        asm("rcp.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 3
        // rcp.approx.ftz
        asm("rcp.approx.ftz.f32 %0, %0;" : "+f"(fv));
#elif MODE == 4
        // ex2.approx (NO .ftz)
        asm("ex2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 5
        // ex2.approx.ftz
        asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(fv));
#elif MODE == 6
        // lg2.approx (NO .ftz)
        asm("lg2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 7
        // lg2.approx.ftz
        asm("lg2.approx.ftz.f32 %0, %0;" : "+f"(fv));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
