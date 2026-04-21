// Does .ftz help SIN/COS/TANH MUFU ops?
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x * 0.01f + 0.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // sin no FTZ
        asm("sin.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 1
        // sin with FTZ
        asm("sin.approx.ftz.f32 %0, %0;" : "+f"(fv));
#elif MODE == 2
        // cos no FTZ
        asm("cos.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 3
        // cos with FTZ
        asm("cos.approx.ftz.f32 %0, %0;" : "+f"(fv));
#elif MODE == 4
        // tanh no FTZ
        asm("tanh.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 5
        // tanh with FTZ
        asm("tanh.approx.ftz.f32 %0, %0;" : "+f"(fv));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
