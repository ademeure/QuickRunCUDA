// __sincosf vs separate sin+cos
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x * 0.01f + (float)u2 * 1e-9f;
    float s = 0.0f, c = 0.0f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Separate sinf + cosf
        s = sinf(fv);
        c = cosf(fv);
        fv = s + c;
#elif MODE == 1
        // __sincosf intrinsic (computes both)
        __sincosf(fv, &s, &c);
        fv = s + c;
#elif MODE == 2
        // PTX sin.approx + cos.approx (2 separate MUFU ops)
        asm("sin.approx.f32 %0, %1;" : "=f"(s) : "f"(fv));
        asm("cos.approx.f32 %0, %1;" : "=f"(c) : "f"(fv));
        fv = s + c;
#elif MODE == 3
        // Just sinf alone (baseline for one call)
        fv = sinf(fv);
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
