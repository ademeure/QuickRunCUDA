// FFMA rounding modes (rn / rz / rm / rp)
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    float fb = 1.000001f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // fma.rn (round-nearest-even, default)
        asm("fma.rn.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 1
        // fma.rz (round-toward-zero)
        asm("fma.rz.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 2
        // fma.rm (round-toward-minus-inf)
        asm("fma.rm.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#elif MODE == 3
        // fma.rp (round-toward-plus-inf)
        asm("fma.rp.f32 %0, %0, %1, %0;" : "+f"(fv) : "f"(fb));
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
