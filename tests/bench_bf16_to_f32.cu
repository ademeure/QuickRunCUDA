// bf16 -> f32 widening cvt
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    unsigned short bv = 0x3F80;
    float fv = (float)threadIdx.x + (float)u2 * 1e-9f;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // PTX cvt.f32.bf16 (widening)
        asm("cvt.f32.bf16 %0, %1;" : "=f"(fv) : "h"(bv));
        bv = (unsigned short)__float_as_uint(fv);  // chain
#elif MODE == 1
        // PTX cvt.f32.f16 (widening)
        asm("cvt.f32.f16 %0, %1;" : "=f"(fv) : "h"(bv));
        bv = (unsigned short)__float_as_uint(fv);
#elif MODE == 2
        // Manual bf16 widening: bv << 16 (no rounding cost)
        unsigned int x = (unsigned int)bv << 16;
        fv = __uint_as_float(x);
        bv = (unsigned short)__float_as_uint(fv);
#elif MODE == 3
        // bitcast __bfloat162float
        fv = (float)__nv_bfloat16(bv);
        bv = (unsigned short)__float_as_uint(fv);
#endif
    }

    if ((int)fv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
