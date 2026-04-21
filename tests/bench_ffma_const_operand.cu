// FFMA with constant operand: reg vs imm vs cmem
#ifndef MODE
#define MODE 0
#endif

extern "C" __constant__ float CMEM_K = 1.0000001f;

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f;
    float fb = 1.0000001f + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // FFMA with all reg operands
        asm("fma.rn.f32 %0, %0, %1, %1;" : "+f"(fv) : "f"(fb));
#elif MODE == 1
        // FFMA with literal immediate constant
        asm("fma.rn.f32 %0, %0, %1, 0.5;" : "+f"(fv) : "f"(fb));
#elif MODE == 2
        // FFMA with cmem operand
        asm("fma.rn.f32 %0, %0, %1, %2;" : "+f"(fv) : "f"(fb), "f"(CMEM_K));
#elif MODE == 3
        // FFMA with all 3 operands as constants
        fv = fv * 1.5f + 0.25f;
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
