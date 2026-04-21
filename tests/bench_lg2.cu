// log2 / lg2 chain latency
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 1.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // PTX lg2.approx (MUFU.LG2)
        asm("lg2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 1
        // CUDA __log2f intrinsic
        fv = __log2f(fv);
#elif MODE == 2
        // log2 via __logf / log(2) — full IEEE
        fv = log2f(fv);
#elif MODE == 3
        // ex2.approx (MUFU.EX2)
        asm("ex2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 4
        // sqrt.approx (MUFU.RSQ then 1/x)
        asm("sqrt.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 5
        // tan.approx
        asm("tanh.approx.f32 %0, %0;" : "+f"(fv));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
