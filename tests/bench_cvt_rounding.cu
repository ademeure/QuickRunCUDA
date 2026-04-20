// Cvt rounding mode throughput
#ifndef MODE
#define MODE 0
#endif

extern "C" __global__ __launch_bounds__(256, 4)
void kernel(float* A, float* B, float* C, int ITERS, int seed, int u2) {
    float fv = (float)threadIdx.x + 0.5f + (float)u2 * 1e-9f;
    int iv = 0;

    for (int i = 0; i < ITERS; i++) {
#if MODE == 0
        // cvt.rni (round-to-nearest-even, default)
        asm("cvt.rni.s32.f32 %0, %1;" : "=r"(iv) : "f"(fv));
#elif MODE == 1
        // cvt.rzi (round-toward-zero / truncate)
        asm("cvt.rzi.s32.f32 %0, %1;" : "=r"(iv) : "f"(fv));
#elif MODE == 2
        // cvt.rmi (round-toward-minus-infinity / floor)
        asm("cvt.rmi.s32.f32 %0, %1;" : "=r"(iv) : "f"(fv));
#elif MODE == 3
        // cvt.rpi (round-toward-plus-infinity / ceil)
        asm("cvt.rpi.s32.f32 %0, %1;" : "=r"(iv) : "f"(fv));
#elif MODE == 4
        // cvt.sat.s32.f32 (saturate to int range)
        asm("cvt.rni.sat.s32.f32 %0, %1;" : "=r"(iv) : "f"(fv));
#endif
        fv += (float)iv * 1e-9f;
    }

    if (iv == seed && (int)fv == seed) ((unsigned*)C)[blockIdx.x * blockDim.x + threadIdx.x] = (unsigned)iv;
}
