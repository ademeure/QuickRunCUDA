// __expf vs ex2.approx vs expf vs ex2*log2(e)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x * 0.01f + 0.5f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // __expf intrinsic
        fv = __expf(fv);
#elif MODE == 1
        // expf standard
        fv = expf(fv);
#elif MODE == 2
        // ex2.approx PTX raw
        asm("ex2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 3
        // ex2.approx.ftz PTX
        asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(fv));
#elif MODE == 4
        // Manual: x = x * 1.4427f (log2(e)), then ex2(x)
        fv = fv * 1.44269504f;
        asm("ex2.approx.f32 %0, %0;" : "+f"(fv));
#elif MODE == 5
        // Manual: ex2(x * log2(e)) with .ftz
        fv = fv * 1.44269504f;
        asm("ex2.approx.ftz.f32 %0, %0;" : "+f"(fv));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
