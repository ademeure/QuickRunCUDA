// Division throughput
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 1.5f + (float)u2 * 1e-9f;
    float fb = 1.000001f + (float)u2 * 1e-9f;
    int iv = (int)threadIdx.x + 1 + (int)u2;
    int ib = 7 + (int)u2;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // Default fp32 div (IEEE compliant)
        fv = fv / fb;
#elif MODE == 1
        // PTX div.approx.f32
        asm("div.approx.f32 %0, %0, %1;" : "+f"(fv) : "f"(fb));
#elif MODE == 2
        // div.full.f32 (full IEEE)
        asm("div.full.f32 %0, %0, %1;" : "+f"(fv) : "f"(fb));
#elif MODE == 3
        // RCP-then-mul (approximation)
        float r;
        asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(fb));
        fv = fv * r;
#elif MODE == 4
        // Integer division
        iv = iv / ib;
#elif MODE == 5
        // Integer modulo
        iv = iv % ib;
#endif
    }

    if ((int)fv == seed && iv == seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)fv + iv;
}
