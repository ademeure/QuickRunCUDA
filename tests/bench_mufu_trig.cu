// MUFU trig variants
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x * 0.1f + 0.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        asm("sin.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 1
        asm("cos.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 2
        asm("tanh.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 3
        asm("ex2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 4
        asm("lg2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 5
        asm("rcp.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 6
        asm("rsqrt.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 7
        asm("sqrt.approx.f32 %0, %0;" : "+f"(fv));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
