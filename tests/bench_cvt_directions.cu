// fp32 <-> s32 cvt symmetry test
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    int iv = (int)threadIdx.x + (int)u2;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // f32 -> s32
        asm("cvt.rni.s32.f32 %0, %1;" : "=r"(iv) : "f"(fv));
        fv = (float)iv;
#elif MODE == 1
        // s32 -> f32
        asm("cvt.rn.f32.s32 %0, %1;" : "=f"(fv) : "r"(iv));
        iv = (int)fv;
#elif MODE == 2
        // f32 -> u32
        unsigned int uv;
        asm("cvt.rni.u32.f32 %0, %1;" : "=r"(uv) : "f"(fv));
        iv = (int)uv;
        fv = (float)uv;
#elif MODE == 3
        // u32 -> f32
        asm("cvt.rn.f32.u32 %0, %1;" : "=f"(fv) : "r"((unsigned)iv));
        iv = (int)fv;
#elif MODE == 4
        // f32 -> f64 (widening)
        double dv;
        asm("cvt.f64.f32 %0, %1;" : "=d"(dv) : "f"(fv));
        fv = (float)dv;
#elif MODE == 5
        // f64 -> f32 (narrowing)
        double dv = (double)fv;
        asm("cvt.rn.f32.f64 %0, %1;" : "=f"(fv) : "d"(dv));
#endif
    }

    if ((int)fv == seed && iv == seed) C[blockIdx.x * blockDim.x + threadIdx.x] = fv;
}
